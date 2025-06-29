import torch
from torch import nn
from torch.nn import functional as F

from .mapping_network import MappingNetowrk, AdaptiveInstanceNorm
from helpers.imle_helpers import get_1x1
from collections import defaultdict


def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers

def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, H, expansion=4, kernel_size=7, use_se=True, reduction=16, dropout=0.0):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-3)
        self.pw_conv1 = nn.Conv2d(dim, expansion * dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.pw_conv2 = nn.Conv2d(expansion * dim, dim, kernel_size=1)

        ## single parameter for residual ratio
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(dim, reduction=reduction)  
        else:
            # Indentity layer if SE is not used
            self.se = nn.Identity()
        self.residual_ratio = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout2d(p=dropout)  # <- NEW LINE

    
    def forward(self, x):
        residual = x
        # Depthwise convolution with larger kernel
        x = self.dw_conv(x)
        # Permute to channels-last for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Permute back to channels-first
        x = x.permute(0, 3, 1, 2)
        # Pointwise conv to expand channels
        x = self.pw_conv1(x)
        x = self.gelu(x)

        # Apply dropout
        x = self.dropout(x)
        # Pointwise conv to compress channels back
        x = self.pw_conv2(x)
        x = self.se(x)
        return x * self.sigmoid(self.residual_ratio) + residual


class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.H = H
        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[res]
        self.adaIN = AdaptiveInstanceNorm(width, H.latent_dim)
        self.resnet = ConvNeXtBlock(width, H, kernel_size=7, 
                                    expansion=H.convnext_expansion, 
                                    use_se=H.use_se,
                                    reduction=H.se_reduction,
                                    dropout=H.dropout_p)

    def forward(self, x, w):
        if self.mixin is not None:
            x = F.interpolate(x, scale_factor=self.base / self.mixin, mode='bicubic')
        x = self.adaIN(x, w)
        x = self.resnet(x)
        return x


class DecBlock2(DecBlock):

    def __init__(self, H, res, mixin, n_blocks, txt_sz):
        super().__init__(H, res, mixin, n_blocks)
        self.adaIN = AdaptiveInstanceNorm(self.widths[res], H.latent_dim + txt_sz)


class Decoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.txt_sz = H.random_proj_sz if H.random_proj_sz > 0 else 512
        self.mapping_network = MappingNetowrk(code_dim=H.latent_dim, n_mlp=H.n_mpl, lr_multiplier=H.mapping_lr_multiplier)
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        first_res = self.resolutions[0]
        self.constant = nn.Parameter(torch.randn(1, self.widths[first_res], first_res, first_res))
        self.resnet = get_1x1(H.width, H.image_channels)
        if self.H.merge_gain:
            self.m_gain = nn.Parameter(torch.ones(H.latent_dim))
        self.gain = nn.Parameter(torch.ones(1, H.image_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.image_channels, 1, 1))
        self.rep_text_emb = H.rep_text_emb
        if self.H.unconditional:
            print("Initializing unconditional model!")
            return
        if H.merge_concat:
            if H.rep_text_emb:
                self.txt_down = nn.ModuleList([nn.Linear(self.txt_sz + H.latent_dim, H.latent_dim) for _ in range(len(blocks))])
            else:
                self.txt_down = nn.Linear(self.txt_sz + H.latent_dim, H.latent_dim)
        else:
            up_dim = 2 * H.latent_dim if H.merge_film else H.latent_dim
            if H.rep_text_emb:
                self.txt_up = nn.ModuleList([nn.Linear(self.txt_sz, up_dim) for _ in range(len(blocks))])
            else:
                self.txt_up = nn.Linear(self.txt_sz, up_dim)
    
    def _merge(self, w, txt_embed, idx=None):
        if self.H.unconditional:
            return w
        if not self.H.merge_concat:
            if idx is not None:
                y = self.txt_up[idx](txt_embed)
            else:
                y = self.txt_up(txt_embed)
            if self.H.merge_film:
                gamma, beta = y.chunk(2, dim=1)
                return gamma * w + beta
        else:
            y = txt_embed
        if self.H.merge_gain:
            nw = self.m_gain * w
        else:
            nw = w
        if self.H.style_gan_merge:
            y = torch.randn_like(y, device=y.device) * y
        if self.H.merge_concat:
            out = torch.cat([nw, y], dim=-1)
            if idx is not None:
                return self.txt_down[idx](out)
            else:
                return self.txt_down(out)
        return y + nw

    def forward(self, latent_code, txt_embed, input_is_w=False):
        assert latent_code.shape[0] == txt_embed.shape[0]
        if self.H.merge_before_map and not self.rep_text_emb:
            w = self._merge(latent_code, txt_embed)
        else:
            w = latent_code
        if not input_is_w:
            w = self.mapping_network(w)
        
        if not self.H.merge_before_map and not self.rep_text_emb:
            w = self._merge(w, txt_embed)
        
        x = self.constant.repeat(latent_code.shape[0], 1, 1, 1)

        for idx, block in enumerate(self.dec_blocks):
            if self.rep_text_emb:
                x = block(x, self._merge(w, txt_embed, idx))
            else:
                x = block(x, w)
        x = self.resnet(x)
        x = self.gain * x + self.bias
        return x


class Decoder2(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.txt_sz = H.random_proj_sz if H.random_proj_sz > 0 else 512
        self.mapping_network = MappingNetowrk(code_dim=H.latent_dim + self.txt_sz, n_mlp=H.n_mpl, lr_multiplier=H.mapping_lr_multiplier)
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock2(H, res, mixin, n_blocks=len(blocks), txt_sz=self.txt_sz))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        first_res = self.resolutions[0]
        self.constant = nn.Parameter(torch.randn(1, self.widths[first_res], first_res, first_res))
        self.resnet = get_1x1(H.width, H.image_channels)
        # if self.H.merge_gain:
        #     self.m_gain = nn.Parameter(torch.ones(H.latent_dim))
        self.gain = nn.Parameter(torch.ones(1, H.image_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.image_channels, 1, 1))

    def forward(self, latent_code, txt_embed, input_is_w=False):
        assert latent_code.shape[0] == txt_embed.shape[0]
        w = torch.cat([latent_code, txt_embed], dim=-1)
        if not input_is_w:
            w = self.mapping_network(w)
        
        x = self.constant.repeat(latent_code.shape[0], 1, 1, 1)

        for idx, block in enumerate(self.dec_blocks):
            x = block(x, w)
        x = self.resnet(x)
        x = self.gain * x + self.bias
        return x


def get_dec(H):
    if H.merge_no_linear:
        return Decoder2(H)
    return Decoder(H)


class IMLE(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.decoder = get_dec(H)

    def forward(self, latents, txt_embed, input_is_w=False):
        return self.decoder.forward(latents, txt_embed, input_is_w)


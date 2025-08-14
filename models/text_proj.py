import torch
from torch import nn
from torch.nn import functional as F


def get_act_text(layer_str):
    if layer_str == "relu":
        return nn.ReLU()
    if layer_str == "leaky":
        return nn.LeakyReLU(0.2)
    if layer_str == "elu":
        return nn.ELU()
    else:
        return nn.ReLU()


class TextProjBlock(nn.Module):
    
    def __init__(self, H, inp_dim, out_dim):
        super().__init__()
        self.H = H
        self.step = H.text_res_step
        act_layer = get_act_text(H.text_act_type)
        if H.num_text_act == 0:
            self.net = nn.Linear(inp_dim, out_dim)
        else:
            nns = [nn.Linear(inp_dim, H.text_hidden_dim), act_layer]
            for _ in range(H.num_text_act - 1):
                nns.append(nn.Linear(H.text_hidden_dim, H.text_hidden_dim))
                nns.append(act_layer)
            nns.append(nn.Linear(H.text_hidden_dim, out_dim))
            if self.step > 0:
                self.net = nn.ModuleList(nns)
            else:
                self.net = nn.Sequential(*nns)
    
    def forward(self, x):
        if self.H.num_text_act == 0 or self.step <= 0:
            return self.net(x)
        
        for idx, block in enumerate(self.net):
            x = block(x)
            if idx == 1:
                res = x
            elif (idx - 1) % (self.step * 2) == 0:
                x = x + res
                res = x  
        return x

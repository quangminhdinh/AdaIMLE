import time
from math import ceil
import numpy as np # type: ignore
import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip

from helpers.utils import is_main_process
from .sampler import Sampler
from torch import autocast


class TextClipCondSamplerV2(Sampler):

    def __init__(self, H, sz, preprocess_fn, kmeans=None):
        super().__init__(H, sz, preprocess_fn)

        print(f"\n{self.__class__.__name__}'s configurations.")
        self.num_latents_per_sample = int(H.force_factor * sz)
        self.pool_size = self.num_latents_per_sample * sz
        print(f"Num latents per sample: {self.num_latents_per_sample}")
        print(f"Pool size: {self.pool_size}")

        self.use_clip = H.use_clip_loss
        clip_model, clip_preprocess = clip.load('ViT-B/32', self.device)
        clip_model = torch.compile(clip_model)
        if self.use_clip:
            self.clip_encoder = clip_model
            self.clip_preprocess = clip_preprocess
            self.clip_preprocess.transforms.pop(2)
            self.clip_preprocess.transforms.pop(2)
            print(f"Adding CLIP loss with coefficient {self.H.clip_coef} and temperature {self.H.clip_temp}!")
        if self.H.use_clip_l2:
            print(f"Applying L2 loss to img clip features with coefficient {self.H.l2_clip_coef}!")
        
        self.sample_texts = ["the petals are purple, the flower is completely open reveling the off red stamen.",
                             "the flower is pink withe petals that are soft, smooth and petals that are separately arranged around sepals in many layers"]
        self.num_rand_samp = H.num_rand_samp
        if(is_main_process()):
            text_input = clip.tokenize(self.sample_texts).to(self.device)
            txt_feats = clip_model.encode_text(text_input).cpu()
            if kmeans is not None:
                labels = kmeans.predict(txt_feats)
                txt_feats = kmeans.centroids[labels]
            self.sample_text_feats = torch.cat([txt.repeat(self.num_rand_samp, 1) for txt in txt_feats]).to(self.device)
            if H.text_unit_norm:
                self.sample_text_feats = F.normalize(self.sample_text_feats, p=2, dim=1).to(self.device)

    def cosine_similarity_loss(self, image_embeds, text_embeds):
        # Normalize the embeddings
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Compute cosine similarity matrix (batch_size x batch_size)
        logits = image_embeds @ text_embeds.T / self.H.clip_temp

        # Labels are positions in the batch
        labels = torch.arange(len(image_embeds), device=image_embeds.device)

        # Contrastive loss (symmetric)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2
    
    def calc_loss(self, inp, tar, text=None, img_clip=None):
        loss = super().calc_loss(inp, tar)
        if self.use_clip and text is not None:
            xhat = (inp + 1.0) * 127.5
            xhat = torch.clamp(xhat, 0.0, 255.0) / 255.0
            inp_xhat = self.clip_preprocess(xhat)
            img_embed = self.clip_encoder.encode_image(inp_xhat)
            loss = loss + self.H.clip_coef * self.cosine_similarity_loss(img_embed, text)
            if self.H.use_clip_l2 and img_clip is not None:
                loss = loss + self.H.l2_clip_coef * self.l2_loss(img_embed, img_clip).mean()
        return loss

    def sample(self, latents, text, gen, snoise=None):
        with torch.no_grad():
            with autocast(device_type='cuda'):
                latents = latents.to(self.device)
                text = text.to(self.device)
                px_z = gen(latents, text, None).permute(0, 2, 3, 1)
                xhat = (px_z + 1.0) * 127.5
                xhat = xhat.detach().cpu().numpy()
                xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
                return xhat
            
    def resample_pool(self, gen, dataset):
        gen.eval()   

        # Determine local pool size
        local_pool_size = self.pool_size // self.world_size
        remainder  = self.pool_size % self.world_size
        if self.rank < remainder:
            local_pool_size = local_pool_size + 1
            local_start = self.rank * local_pool_size
        else:
            local_start = self.rank * local_pool_size + remainder

        samp_idxs = []
        for i in range(local_start // self.num_latents_per_sample, 
                       (local_start + local_pool_size) // self.num_latents_per_sample + 1):
            start = max(local_start, i * self.num_latents_per_sample)
            end = min(local_start + local_pool_size, (i + 1) * self.num_latents_per_sample)
            samp_idxs.append(torch.ones(end - start, dtype=torch.int) * i)
        samp_idxs = torch.cat(samp_idxs)
        assert samp_idxs.shape[0] == local_pool_size
        # if(is_main_process()):
        #     print(f"Sample indices: {samp_idxs[-20:].cpu()}")

        # Generate local pool latents and prepare container for projected features
        local_pool_latents = torch.randn((local_pool_size, self.H.latent_dim), 
                                         device=self.device, 
                                         generator=self.generator_seed)
        # Assuming pool_samples_proj is preallocated with shape (self.pool_size, projection_dim)

        local_pool_proj = torch.empty((local_pool_size, self.dci_dim), device=self.device)

        # Process local chunk in batches
        num_batch = ceil(local_pool_size / self.H.imle_batch)
        # for j in tqdm(range(num_batch), desc="Resample:", disable=(not is_main_process())):
        for j in range(num_batch):
            batch_slice = slice(j * self.H.imle_batch, 
                                min((j + 1) * self.H.imle_batch, local_pool_size))
            cur_latents = local_pool_latents[batch_slice]
            cur_txt = dataset.txt_clip[samp_idxs[batch_slice]]
            with torch.no_grad():
                with autocast(device_type='cuda'):
                    outputs = gen(cur_latents, cur_txt, None)
                    if self.H.search_type == 'lpips':
                        proj = self.get_projected(outputs, False)
                    elif self.H.search_type == 'l2':
                        proj = self.get_l2_feature(outputs, False)
                    elif self.H.search_type == 'combined':
                        proj = self.get_combined_feature(outputs, False)
                    else:
                        proj = self.get_combined_feature(outputs, False)
                    local_pool_proj[batch_slice] = proj

        torch.distributed.barrier()

        gathered_latents = [torch.empty_like(local_pool_latents) for _ in range(self.world_size)]
        gathered_proj = [torch.empty_like(local_pool_proj) for _ in range(self.world_size)]

        torch.distributed.all_gather(gathered_latents, local_pool_latents)
        torch.distributed.all_gather(gathered_proj, local_pool_proj)

        gen.train()

        torch.distributed.barrier()

        # Aggregate the full pool latents and projected features
        self.pool_latents = torch.cat(gathered_latents, dim=0).to('cpu')
        self.pool_samples_proj = torch.cat(gathered_proj, dim=0).to('cpu').numpy().astype(np.float32)

    def imle_sample_force(self, dataset, gen, to_update=None):
        """
        Optimized force resampling routine using FAISS for batched nearest-neighbor search.
        In a DDP setting, each process handles a different subset of the dataset features,
        performs NN search locally, and then the results are merged and broadcast so that
        all processes end up with the complete global results.
        """
        if is_main_process():
            t1 = time.time()
            print("Starting pool resampling...")

        self.resample_pool(gen, dataset)
        torch.distributed.barrier()  # Ensure all processes complete the pool resample

        if(is_main_process()):
            print(f"Resampling pool took {time.time() - t1:.2f} seconds")
        
        torch.cuda.empty_cache()

        self.selected_dists_tmp[:] = np.inf

        with torch.no_grad():
            # Total number of dataset samples.
            total_datapoints = self.dataset_proj.shape[0]

            # --------------------
            # Partition the dataset features so each process works on a different chunk.
            chunk_size = total_datapoints // self.world_size
            remainder = total_datapoints % self.world_size
            if self.rank < remainder:
                local_size = chunk_size + 1
                local_start = self.rank * local_size
            else:
                local_size = chunk_size
                local_start = self.rank * local_size + remainder
            local_end = min(local_start + local_size, self.sz)

            # Prepare local updated arrays.
            local_updated_dists = self.selected_dists_tmp[local_start:local_end].clone()
            local_updated_latents = self.selected_latents_tmp[local_start:local_end].clone()

            # for samp_idx in tqdm(range(local_start, local_end), desc="IMLE:", disable=(not is_main_process())):
            for samp_idx in range(local_start, local_end):
                pool_feats = self.pool_samples_proj[samp_idx * self.num_latents_per_sample :
                                                    (samp_idx + 1) * self.num_latents_per_sample]
                self.gpu_index_flat.add(pool_feats)
                distances, indices = self.gpu_index_flat.search(self.dataset_proj[samp_idx][np.newaxis, ...], 1)
                distance = torch.from_numpy(distances).squeeze()  # (local_size,)
                ind = torch.from_numpy(indices).squeeze()
                self.gpu_index_flat.reset()
                if distance < self.selected_dists_tmp[samp_idx]:
                    local_updated_dists[samp_idx - local_start] = distance
                    local_updated_latents[samp_idx - local_start] = \
                        self.pool_latents[samp_idx * self.num_latents_per_sample + ind]
                
            if is_main_process():
                gathered_dists = [None for _ in range(self.world_size)]
                gathered_latents = [None for _ in range(self.world_size)]
            else:
                gathered_dists = None
                gathered_latents = None

            torch.distributed.gather_object(local_updated_dists, gathered_dists, dst=0)
            torch.distributed.gather_object(local_updated_latents, gathered_latents, dst=0)

            torch.distributed.barrier()  # Ensure all processes complete the gather

            if is_main_process():
                full_updated_dists = torch.cat(gathered_dists, dim=0).to(self.device)
                full_updated_latents = torch.cat(gathered_latents, dim=0).to(self.device)
                perturbation = self.H.imle_perturb_coef * torch.randn(
                    (self.sz, self.H.latent_dim), 
                    device=self.device,
                    generator=self.generator_seed)
                full_updated_latents += perturbation
            else:
                full_updated_dists = torch.empty(self.sz, dtype=torch.float32, device=self.device)
                full_updated_latents = torch.empty(self.sz, self.H.latent_dim, dtype=torch.float32, device=self.device)

            torch.distributed.barrier()

            torch.distributed.broadcast(full_updated_dists, src=0)
            torch.distributed.broadcast(full_updated_latents, src=0)

            torch.distributed.barrier()


            # Move the broadcasted results to CPU if desired.
            self.selected_dists_tmp = full_updated_dists.cpu()
            self.selected_latents_tmp = full_updated_latents.cpu()

            # Update last and current selected latents on all processes.
            self.last_selected_latents = self.selected_latents.clone()
            self.selected_latents = self.selected_latents_tmp.clone()

            if is_main_process():
                print(f"Force resampling took {time.time() - t1:.2f} seconds")

        torch.distributed.barrier()  # Ensure synchronization before leaving the function

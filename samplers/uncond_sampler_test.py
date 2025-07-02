import time
from math import ceil
import numpy as np # type: ignore
import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip

from helpers.utils import is_main_process
from .text_clip_cond_sampler_v2 import TextClipCondSamplerV2
from torch import autocast


class UncondSamplerTest(TextClipCondSamplerV2):

    def __init__(self, H, sz, preprocess_fn, kmeans=None, rand_proj=None):
        super().__init__(H, sz, preprocess_fn, kmeans, rand_proj)
        
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
            self.gpu_index_flat.add(self.pool_samples_proj)
            # for samp_idx in tqdm(range(local_start, local_end), desc="IMLE:", disable=(not is_main_process())):
            all_distances, all_indices = self.gpu_index_flat.search(self.dataset_proj[local_start:local_end], 1)
            for samp_idx in range(local_start, local_end):
                distance = torch.from_numpy(all_distances[samp_idx]).squeeze()  # (local_size,)
                ind = torch.from_numpy(all_indices[samp_idx]).squeeze()
                if distance < self.selected_dists_tmp[samp_idx]:
                    local_updated_dists[samp_idx - local_start] = distance
                    local_updated_latents[samp_idx - local_start] = self.pool_latents[ind]
                
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
            
            self.gpu_index_flat.reset()

            if is_main_process():
                print(f"Force resampling took {time.time() - t1:.2f} seconds")

        torch.distributed.barrier()  # Ensure synchronization before leaving the function

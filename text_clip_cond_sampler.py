from math import ceil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from helpers.utils import is_main_process
from sampler import Sampler
from torch import autocast


class TextClipCondSampler(Sampler):

    def __init__(self, H, sz, preprocess_fn):
        super().__init__(H, sz, preprocess_fn)

        print(f"\n{self.__class__.__name__}'s configurations.")
        self.pool_size = int(H.force_factor * sz)
        self.dataset_proj = torch.empty([sz, self.dci_dim], dtype=torch.float32, device=self.device)
        print(f"Pool size: {self.pool_size}")
        
    def init_projection(self, dataset):

        for ind, x in enumerate(DataLoader(TensorDataset(dataset), batch_size=self.H.n_batch)):
            batch_slice = slice(ind * self.H.n_batch, ind * self.H.n_batch + x[0].shape[0])
            if(self.H.search_type == 'lpips'):
                self.dataset_proj[batch_slice] = self.get_projected(self.preprocess_fn(x)[1])
            elif(self.H.search_type == 'l2'):
                self.dataset_proj[batch_slice] = self.get_l2_feature(self.preprocess_fn(x)[1])
            # elif(self.H.search_type == 'vae'):
            #     self.dataset_proj[batch_slice] = self.get_vae_features(self.preprocess_fn(x)[1]).cpu()
            elif(self.H.search_type == 'combined'):
                self.dataset_proj[batch_slice] = self.get_combined_feature(self.preprocess_fn(x)[1])
            else:
                exit()

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

    def find_min_latent(self, gen, txt_clip, sample_proj, log=False):
        gen.eval()   

        # Generate local pool latents and prepare container for projected features
        local_pool_latents = torch.randn((self.pool_size, self.H.latent_dim), 
                                         device=self.device, 
                                         generator=self.generator_seed)
        # Assuming pool_samples_proj is preallocated with shape (self.pool_size, projection_dim)

        local_pool_proj = torch.empty((self.pool_size, self.dci_dim), device=self.device)

        # Process local chunk in batches
        num_batch = self.pool_size // self.H.imle_batch
        total_time = 0
        for j in range(num_batch):
            start = time.time()
            batch_slice = slice(j * self.H.imle_batch, (j + 1) * self.H.imle_batch)
            cur_latents = local_pool_latents[batch_slice]
            with torch.no_grad():
                with autocast(device_type='cuda'):
                    outputs = gen(cur_latents, txt_clip.repeat(cur_latents.shape[0], 1), None)
                    if self.H.search_type == 'lpips':
                        proj = self.get_projected(outputs, False)
                    elif self.H.search_type == 'l2':
                        proj = self.get_l2_feature(outputs, False)
                    elif self.H.search_type == 'combined':
                        proj = self.get_combined_feature(outputs, False)
                    else:
                        proj = self.get_combined_feature(outputs, False)
                    local_pool_proj[batch_slice] = proj
            total_time += time.time() - start
        total_time /= num_batch

        # self.gpu_index_flat.add(local_pool_proj.cpu().numpy().astype(np.float32))  # add entire pool

        # distances, indices = self.gpu_index_flat.search(sample_proj, 1)
        # local_distances = torch.from_numpy(distances).squeeze()
        # local_indices   = torch.from_numpy(indices).squeeze() sz x dci_dim
        # self.gpu_index_flat.reset()
        start = time.time()
        all_dists = torch.sqrt(torch.sum((sample_proj - local_pool_proj) ** 2, dim=-1))
        ind = torch.argmin(all_dists)
        knn_time = time.time() - start

        if is_main_process() and log:
            print(f"Average projection calculation time: {total_time}")
            print(f"KNN calculation time: {knn_time}")

        gen.train()

        return all_dists[ind], local_pool_latents[ind]

    def imle_sample_force(self, dataset, gen, to_update=None):
        """
        Optimized force resampling routine using FAISS for batched nearest-neighbor search.
        In a DDP setting, each process handles a different subset of the dataset features,
        performs NN search locally, and then the results are merged and broadcast so that
        all processes end up with the complete global results.
        """
        if is_main_process():
            t1 = time.time()
            print("Starting force imle resampling...")
        
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

            for samp_idx in tqdm(range(local_start, local_end), desc="IMLE:", disable=(not is_main_process())):
                distance, latent = self.find_min_latent(
                    gen, dataset.txt_clip[samp_idx].unsqueeze(0), 
                    # self.dataset_proj[samp_idx][np.newaxis, ...]
                    self.dataset_proj[samp_idx].unsqueeze(0),
                    log=(samp_idx == local_start)
                )
                if distance < self.selected_dists_tmp[samp_idx]:
                    local_updated_dists[samp_idx - local_start] = distance
                    local_updated_latents[samp_idx - local_start] = latent
                
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

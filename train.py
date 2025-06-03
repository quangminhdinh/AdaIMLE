import os
import time

# from comet_ml import Experiment, ExistingExperiment
import torch
from torch.utils.data.distributed import DistributedSampler
from cleanfid import fid
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import wandb
from dataloaders.data import set_up_data
from helpers.train_helpers import (load_imle, load_opt, save_model, set_up_hyperparams, update_ema)
from helpers.utils import init_distributed_mode, is_main_process, get_world_size, get_rank
from samplers import Sampler, TextClipCondSamplerV2
from visual.interpolate import random_interp
from visual.utils import (generate_and_save, generate_for_NN, generate_for_NN_wtext,
                          generate_visualization, generate_visualization_wtext,
                          get_sample_for_visualization, generate_and_save_wtext)
from helpers.improved_precision_recall import compute_prec_recall
from dataloaders import text_clip_cond_collate, ZippedDataset
from torch import autocast
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch.distributed as dist
from tqdm import tqdm

def isValid(num):
    return not num != num

def cleanup():
    dist.destroy_process_group()

def print_seed(device):
    cpu_seed = torch.initial_seed()
    cuda_seed = torch.cuda.initial_seed()
    print(f"Device {device} CPU seed = {cpu_seed}, GPU seed = {cuda_seed} \n")

def training_step_imle(H, n, targets, latents, text, imle, ema_imle, optimizer, loss_fn, scaler, clip_feat=None):
    targets_permuted = targets.permute(0, 3, 1, 2)
    with autocast(device_type='cuda'):
        px_z = imle(latents, text)
        loss = loss_fn(px_z, targets.permute(0, 3, 1, 2), text=(text if H.use_clip_loss else None),
                       img_clip=clip_feat)
        loss_measure = loss.clone()
        num_resolutions = 1

        if(H.use_multi_res):
            
            for scale in H['multi_res_scales']:
                px_z_scale = F.interpolate(px_z, size=(scale,scale), antialias=True, mode='bicubic')
                targets_scale = F.interpolate(targets_permuted, size=(scale,scale), antialias=True, mode='bicubic')
                loss_scale = loss_fn(px_z_scale, targets_scale, text=(text if H.use_clip_loss_multi_res else None)
                                    # ,img_clip=(clip_feat if H.use_clip_loss_multi_res else None)
                                     )
                
                loss.add_(loss_scale)
                num_resolutions += 1

    loss = loss / num_resolutions
    loss = loss / (H.accumulation_steps)
    
    scaler.scale(loss).backward()
    return loss_measure.detach()

def train_loop_imle(H, data_train, data_valid, preprocess_fn, imle, ema_imle, logprint, experiment=None):
    if not H.use_text:
        subset_len = len(data_train)
        if H.subset_len != -1:
            subset_len = H.subset_len
        for data_train in DataLoader(data_train, batch_size=subset_len):
            data_train = TensorDataset(data_train[0])
            break

    optimizer, scheduler, scaler, best_fid, iterate, starting_epoch = load_opt(H, imle, logprint)

    H.ema_rate = torch.as_tensor(H.ema_rate)

    subset_len = H.subset_len if H.subset_len != -1 else len(data_train)

    if H.use_text:
        sampler = TextClipCondSamplerV2(H, subset_len, preprocess_fn)
    else:
        sampler = Sampler(H, subset_len, preprocess_fn)
    torch.distributed.barrier()
    device = torch.device("cuda", torch.cuda.current_device())

    epoch = starting_epoch 

    split_x_tensor = data_train.trX if H.use_text else data_train.tensors[0]
    split_x = TensorDataset(split_x_tensor)

    sampler.init_projection(split_x_tensor)
    
    torch.distributed.barrier()

    viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn, H.num_images_visualize, H.dataset)

    latent_for_visualization = []

    if(is_main_process()):
        latent_for_visualization = torch.randn(H.num_rows_visualize, H.num_images_visualize, H.latent_dim).to(device)
    
    mean_loss = float('inf')
    metrics = {
        'mean_loss': mean_loss
    }
        
    while (epoch < H.num_epochs):

        torch.distributed.barrier()

        # Update the IMLE force resampling every imle_force_resample epochs.
        if epoch % H.imle_force_resample == 0:
            sampler.imle_sample_force(data_train, imle)

        torch.distributed.barrier()

        if (epoch % 20 == 0 and is_main_process()):
            latents = sampler.selected_latents[:H.num_images_visualize]
            with torch.no_grad():
                imle.eval()
                if H.use_text:
                    generate_for_NN_wtext(sampler, split_x_tensor, latents, data_train.txt_clip, 
                                data_train.txt_list, viz_batch_original.shape, imle,
                                f'{H.save_dir}/NN-samples_{epoch}-imle', logprint)
                else:
                    generate_for_NN(sampler, split_x_tensor[:H.num_images_visualize], latents,
                                viz_batch_original.shape, imle,
                                f'{H.save_dir}/NN-samples_{epoch}-imle.png', logprint)
                imle.train()

        # Create a dataset that pairs images with their current latents.
        torch.distributed.barrier()
        if H.use_text:
            data_train.update_latent(sampler.selected_latents)
            comb_dataset = data_train
        else:
            comb_dataset = ZippedDataset(split_x, TensorDataset(sampler.selected_latents))

        # Use a DistributedSampler if in distributed training.
        train_sampler = DistributedSampler(comb_dataset, 
                                           shuffle=True, 
                                           num_replicas=H.world_size,
                                           rank=H.local_rank,
                                           seed=H.seed)
        
        data_loader = DataLoader(comb_dataset, batch_size=H.n_batch, sampler=train_sampler,
                                    pin_memory=True, num_workers=4, 
                                    persistent_workers=True, 
                                    multiprocessing_context="spawn",
                                    shuffle=False,
                                    collate_fn=text_clip_cond_collate if H.use_text else None)

        # If using distributed sampler, set the epoch for shuffling
        train_sampler.set_epoch(epoch)

        if(is_main_process()):
            start_time = time.time()

        torch.distributed.barrier()
        # Main training loop.

        epoch_loss_sum = 0.0  # We'll accumulate loss from each batch.
        epoch_iter_count = 0
        accum_counter = 0
        imle.zero_grad(set_to_none=True)

        # for data in tqdm(data_loader, desc=f"Epoch {epoch}/{H.num_epochs}:", disable=(not is_main_process())):
        for data in data_loader:
            _, target = preprocess_fn([data["raw_img"]])
            target = target.to(device)
            latents = data["latent"].to(device)
            text = data["text"].to(device)
            if H.use_clip_loss and H.use_clip_l2:
                img_feat = data["img"].to(device)
            else:
                img_feat = None

            loss = training_step_imle(H, target.shape[0], target, latents, text, imle, ema_imle,
                               optimizer, sampler.calc_loss, scaler, clip_feat=img_feat)
            
            epoch_loss_sum += loss.item()
            epoch_iter_count += 1

            accum_counter += 1

            # When we have accumulated enough mini-batches, perform the step.
            if accum_counter % H.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                imle.zero_grad(set_to_none=True)
                update_ema(imle.module, ema_imle, H.ema_rate)
            
            if iterate % H.iters_per_images == 0:
                if(is_main_process()):
                    imle.eval()
                    with torch.no_grad():
                        if H.use_text:
                            generate_visualization_wtext(H, sampler, viz_batch_original, data_train.txt_clip, data_train.txt_list,
                                                sampler.selected_latents[0: H.num_images_visualize],
                                                sampler.last_selected_latents[0: H.num_images_visualize],
                                                latent_for_visualization,
                                                viz_batch_original.shape, imle,
                                                f'{H.save_dir}/samples-{iterate}', logprint, experiment)
                        else:
                            generate_visualization(H, sampler, viz_batch_original,
                                                sampler.selected_latents[0: H.num_images_visualize],
                                                sampler.last_selected_latents[0: H.num_images_visualize],
                                                latent_for_visualization,
                                                viz_batch_original.shape, imle,
                                                f'{H.save_dir}/samples-{iterate}.png', logprint, experiment)
                    imle.train()
            iterate += 1
            
            if iterate % H.iters_per_ckpt == 0 and is_main_process():
                fp = os.path.join(H.save_dir, f'iter-{iterate}')
                logprint(f'Saving model@ {iterate} to {fp}')
                save_model(fp, imle, ema_imle, optimizer, scheduler, scaler, H)
            torch.distributed.barrier()

        
        if accum_counter % H.accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            imle.zero_grad(set_to_none=True)
            update_ema(imle.module, ema_imle, H.ema_rate)
        
        epoch_loss_tensor = torch.tensor(epoch_loss_sum, device=device)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        total_batches_tensor = torch.tensor(epoch_iter_count, device=device)
        dist.all_reduce(total_batches_tensor, op=dist.ReduceOp.SUM)

        mean_loss = epoch_loss_tensor.item() / total_batches_tensor.item()
        
        metrics = {
            'mean_loss': mean_loss,
            'curr_lr': optimizer.param_groups[0]['lr'],
        }

        if (epoch > 0 and epoch % H.fid_freq == 0):
            torch.cuda.empty_cache()
            if H.use_text:
                generate_and_save_wtext(H, imle, sampler, data_train.txt_clip, min(5000, subset_len * H.fid_factor))
            else:
                generate_and_save(H, imle, sampler, min(5000, subset_len * H.fid_factor))
            torch.distributed.barrier()
            torch.cuda.empty_cache()
            if(is_main_process()):
                cur_fid = fid.compute_fid(f'{H.data_root}/img', f'{H.save_dir}/fid/', verbose=False, use_dataparallel=False, num_workers=0, device=device)
                
                precision, recall = compute_prec_recall(f'{H.data_root}/img', f'{H.save_dir}/fid/')
                if cur_fid < best_fid:
                    best_fid = cur_fid
                
                metrics.update({'fid': cur_fid, 'best_fid': best_fid, 'precision': precision, 'recall': recall})

                if cur_fid == best_fid:
                    fp = os.path.join(H.save_dir, 'best_fid')
                    logprint(f'Saving model best fid {best_fid} @ {iterate} to {fp}')
                    logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **metrics)
                    save_model(fp, imle, ema_imle, optimizer, scheduler, scaler, H)

            torch.distributed.barrier()


        if(is_main_process()):
            print(f'Epoch {epoch} took {time.time() - start_time} seconds')

            if epoch % 5 == 0:
                logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **metrics)


        if (epoch % 25 == 0 and is_main_process()):
            imle.eval()
            with torch.no_grad():
                if H.use_text:
                    generate_visualization_wtext(H, sampler, viz_batch_original, data_train.txt_clip, data_train.txt_list,
                                        sampler.selected_latents[0: H.num_images_visualize],
                                        sampler.last_selected_latents[0: H.num_images_visualize],
                                        latent_for_visualization,
                                        viz_batch_original.shape, imle,
                                        f'{H.save_dir}/latest', logprint, experiment)
                else:
                    generate_visualization(H, sampler, viz_batch_original,
                                        sampler.selected_latents[0: H.num_images_visualize],
                                        sampler.last_selected_latents[0: H.num_images_visualize],
                                        latent_for_visualization,
                                        viz_batch_original.shape, imle,
                                        f'{H.save_dir}/latest.png', logprint, experiment)
            imle.train()

        if is_main_process():
            if H.use_wandb:
                wandb.log(metrics, step=iterate)

        if (epoch % 5 == 0 and experiment is not None and is_main_process()):
            experiment.log_metrics(metrics, epoch=epoch, step=iterate)
        
        if epoch % H.epoch_per_save == 0 and is_main_process() and isValid(mean_loss):
            fp = os.path.join(H.save_dir, 'latest')
            logprint(f'Saving latest model@ {iterate} to {fp}')
            save_model(fp, imle, ema_imle, optimizer, scheduler, scaler, H)
        torch.distributed.barrier()

        epoch += 1
    
    if is_main_process():
        print("Training complete. Saving final model.")
        fp = os.path.join(H.save_dir, 'final')
        logprint(f'Saving final model@ {iterate} to {fp}')
        save_model(fp, imle, ema_imle, optimizer, scheduler, scaler, H)
    torch.distributed.barrier()


def main():
    init_distributed_mode()
    
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)

    H.world_size = get_world_size()
    H.local_rank = get_rank()
    # imle, ema_imle = load_imle(H, logprint)

    experiment = None
    if(is_main_process()):
        if H.use_wandb:
            wandb.init(
                name=H.wandb_name,
                project=H.wandb_project,
                config=H,
                mode=H.wandb_mode,
                id=H.wandb_id,
                resume="allow" if H.wandb_id else None
            )
        # print(H)
        if False and H.use_comet and H.comet_api_key:
            if(H.comet_experiment_key):
                print("Resuming experiment")
                experiment = ExistingExperiment(
                    api_key=H.comet_api_key,
                    previous_experiment=H.comet_experiment_key
                )
                experiment.log_parameters(H)

            else:
                experiment = Experiment(
                    api_key=H.comet_api_key,
                    project_name="adaptiveimle-ablation",
                    workspace="serchirag",
                )
                experiment.set_name(H.comet_name)
                experiment.log_parameters(H)
        else:
            experiment = None

        os.makedirs(f'{H.save_dir}/fid', exist_ok=True)

    torch.distributed.barrier()

    if(is_main_process()):
        logprint('training model', H.desc, 'on', H.dataset)

    imle, ema_imle = load_imle(H, logprint)

    if(is_main_process()):
        num_params = sum(p.numel() for p in imle.parameters())
        print("Number of parameters in IMLE: ", num_params)
        logprint("Number of parameters in IMLE: ", num_params)
        H.num_params = num_params
        if(experiment is not None):
            experiment.log_parameter("num_params", num_params)

    if(H.mode == 'train'):

        train_loop_imle(H, data_train, data_valid_or_test, preprocess_fn, imle, ema_imle, logprint, experiment)

    elif H.mode == 'eval_fid':
        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        sampler = Sampler(H, len(data_train), preprocess_fn)
        # generate_and_save(H, imle, sampler, 5000)
        torch.distributed.barrier()
        
        if(is_main_process()):
            print("Generating samples for FID")

        imle.eval()
        generate_and_save(H, imle, sampler, 50000)
        torch.distributed.barrier()
        # if(is_main_process()):
            
        #     cur_fid = fid.compute_fid(f'{H.data_root}/img', f'{H.save_dir}/fid/', verbose=False)
        #     print("FID: ", cur_fid)

    elif H.mode == 'interpolate':
        if(is_main_process()):
            print("Generating interpolations")
            os.makedirs(f'{H.save_dir}/interp', exist_ok=True)

        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        
        imle.eval()
        with torch.no_grad():
            sampler = Sampler(H, subset_len, preprocess_fn)
            torch.distributed.barrier()

            rank = get_rank()
            world_size = get_world_size()
            for i in range(rank,H.num_images_to_generate, world_size):
                random_interp(H, sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/interp/{i}.png', logprint)
                
    cleanup()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

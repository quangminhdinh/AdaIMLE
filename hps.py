# from distutils.util import strtobool

HPARAMS_REGISTRY = {}


class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

cifar10 = Hyperparams()
cifar10.width = 384
cifar10.lr = 0.0002
cifar10.wd = 0.01
cifar10.dec_blocks = "1x1,4m1,4x8,8m4,8x16,16m8,16x16,32m16,32x21"
cifar10.warmup_iters = 100
cifar10.dataset = 'cifar10'
cifar10.n_batch = 16
cifar10.imle_batch = 32 
cifar10.ema_rate = 0.9999
cifar10.l2_search_downsample = 1.0
cifar10.multi_res_scales = '8,12,16,24,28'
cifar10.convnext_expansion = 6
HPARAMS_REGISTRY['cifar10'] = cifar10


stl10 = Hyperparams()
stl10.width = 384
stl10.lr = 0.0002
stl10.wd = 0.01
stl10.dec_blocks = "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12"
# stl10.dec_blocks = "1x1,4m1,4x8,8m4,8x10,16m8,16x10,32m16,32x10,64m32,64x10"

stl10.warmup_iters = 100
stl10.dataset = 'stl10'
stl10.n_batch = 8
stl10.imle_batch = 32 
stl10.ema_rate = 0.9999
stl10.l2_search_downsample = 0.5
stl10.multi_res_scales = '16,32,48'
stl10.convnext_expansion = 4
HPARAMS_REGISTRY['stl10'] = stl10


fewshot = Hyperparams()
fewshot.width = 384
fewshot.lr = 0.0002
fewshot.wd = 0.01
fewshot.dec_blocks = '1x4,4m1,4x4,8m4,8x4,16m8,16x3,32m16,32x2,64m32,64x2,128m64,128x2,256m128'
# fewshot.dec_blocks = '1x4,2m1,2x2,4m2,4x4,8m4,8x4,16m8,16x3,32m16,32x2,64m32,64x2,128m64,128x2,256m128'
fewshot.warmup_iters = 10
fewshot.dataset = 'fewshot'
fewshot.n_batch = 4
fewshot.use_text = False
fewshot.rep_text_emb = False
# fewshot.style_gan_merge = False
# fewshot.use_clip_loss = False
fewshot.ema_rate = 0.9999
fewshot.l2_search_downsample = 0.125
fewshot.multi_res_scales = '8,12,16,24,32,48,64,96,128,150,200,230'
HPARAMS_REGISTRY['fewshot'] = fewshot

def parse_args_and_update_hparams(H, parser, s=None):
    args = parser.parse_args(s)
    valid_args = set(args.__dict__.keys())
    hparam_sets = [x for x in args.hparam_sets.split(',') if x]
    for hp_set in hparam_sets:
        hps = HPARAMS_REGISTRY[hp_set]
        for k in hps:
            if k not in valid_args:
                raise ValueError(f"{k} not in default args")
        parser.set_defaults(**hps)
    H.update(parser.parse_args(s).__dict__)

    try:
        value = H['multi_res_scales']
        list_value = value.split(',')
        list_value_int = [int(x) for x in list_value]
        H['multi_res_scales'] = list_value_int
    except:
        pass

def add_imle_arguments(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--data_root', type=str, default='./datasets/ffhq/')
    parser.add_argument('--desc', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='cifar10')  # path to dataset
    parser.add_argument('--hparam_sets', '--hps', type=str)  # e.g. 'fewshot'
    parser.add_argument('--enc_blocks', type=str, default=None)  # specify encoder blocks, e.g. '1x2,4m1,4x4,8m4,8x5,16m8,16x8,32m16,32x5,64m32,64x4,128m64,128x4,256m128'
    parser.add_argument('--dec_blocks', type=str, default=None)  # specify decoder blocks, e.g. '256x4,128m64,128x4,64m32,64x4,32m16,32x5,16m8,16x8,8m4,8x5,4m1,4x4,1x2'
    parser.add_argument('--width', type=int, default=512)  # width of encoder and decoder convs
    parser.add_argument('--custom_width_str', type=str, default='')  # custom width for each block
    parser.add_argument('--bottleneck_multiple', type=float, default=0.25)  # coefficient width of bottleneck layers, e.g. 0.25 means 1/4 of width

    parser.add_argument('--restore_path', type=str, default=None)  # restore from checkpoint
    parser.add_argument('--restore_ema_path', type=str, default=None)  # restore ema from checkpoint
    parser.add_argument('--restore_log_path', type=str, default=None)  # restore log from checkpoint
    parser.add_argument('--restore_optimizer_path', type=str, default=None)  # restore optimizer from checkpoint
    parser.add_argument('--restore_scheduler_path', type=str, default=None)  # restore optimizer from scheduler
    parser.add_argument('--restore_scaler_path', type=str, default=None)  # restore optimizer from scheduler

    parser.add_argument('--restore_latent_path', type=str, default=None)  # restore nearest neighbour latent codes from checkpoint
    parser.add_argument('--restore_threshold_path', type=str, default=None)  # restore nearest neighbour thresholds, i.e., \tau_i, from checkpoint
    parser.add_argument('--ema_rate', type=float, default=0.999)  # exponential moving average rate
    parser.add_argument('--warmup_iters', type=float, default=0)  # number of iterations for warmup for scheduler
    parser.add_argument('--lr_decay_iters', type=float, default=4000)  # number of iterations for warmup for scheduler
    parser.add_argument('--lr_decay_rate', type=float, default=0.25)  # number of iterations for warmup for scheduler
    parser.add_argument('--mapping_lr_multiplier', type=float, default=1.00)  # weight decay

    parser.add_argument('--compile', default=True, type=lambda x: bool(x))  # whether to use nearest neighbour search

    parser.add_argument('--lr', type=float, default=0.00015)  # learning rate
    parser.add_argument('--lr2', type=float, default=0.00005)  # learning rate

    parser.add_argument('--wd', type=float, default=0.00)  # weight decay
    parser.add_argument('--num_epochs', type=int, default=10000)  # number of epochs
    parser.add_argument('--n_batch', type=int, default=4)  # batch size
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.9)
    parser.add_argument('--adam_eps', type=float, default=1e-8)

    parser.add_argument('--iters_per_ckpt', type=int, default=5000)  # number of iterations per checkpoint
    parser.add_argument('--iters_per_save', type=int, default=1000)  # number of iterations per saving the latest models
    parser.add_argument('--epoch_per_save', type=int, default=50)  # number of epochs per saving the latest models
    parser.add_argument('--iters_per_images', type=int, default=1000)  # number of iterations per sample save
    parser.add_argument('--num_images_visualize', type=int, default=10)  # number of images to visualize
    parser.add_argument('--num_rows_visualize', type=int, default=9)  # number of rows to visualize, e.g. 3 means 3x8=24 images
    parser.add_argument('--merge_no_linear', default=False, type=lambda x: bool(x))
    parser.add_argument('--text_unit_norm', default=False, type=lambda x: bool(x))
    parser.add_argument('--random_proj_sz', type=int, default=-1)
    parser.add_argument('--text_noise_ratio', type=float, default=0.1)
    parser.add_argument('--normalize_random_proj', default=False, type=lambda x: bool(x))
    parser.add_argument('--unconditional', default=False, type=lambda x: bool(x))

    parser.add_argument('--residual_ratio', type=float, default=1.0)
    parser.add_argument('--use_text', default=False, type=lambda x: bool(x))
    parser.add_argument('--rep_text_emb', default=False, type=lambda x: bool(x))
    parser.add_argument('--style_gan_merge', default=False, type=lambda x: bool(x))
    parser.add_argument('--merge_gain', default=False, type=lambda x: bool(x))
    parser.add_argument('--merge_concat', default=False, type=lambda x: bool(x))
    parser.add_argument('--merge_film', default=False, type=lambda x: bool(x))
    parser.add_argument('--use_clip_loss', default=False, type=lambda x: bool(x))
    parser.add_argument('--use_clip_loss_multi_res', default=False, type=lambda x: bool(x))
    parser.add_argument('--use_clip_l2', default=False, type=lambda x: bool(x))
    parser.add_argument('--l2_clip_coef', type=float, default=0.1)
    parser.add_argument('--merge_before_map', default=False, type=lambda x: bool(x))
    parser.add_argument('--num_rand_samp', type=int, default=5)  # dci number of components
    parser.add_argument('--n_clusters', type=int, default=-1)

    parser.add_argument('--accumulation_steps', type=int, default=1)  # accumulation steps
    parser.add_argument('--num_comp_indices', type=int, default=2)  # dci number of components
    parser.add_argument('--num_simp_indices', type=int, default=7)  # dci number of simplices
    parser.add_argument('--imle_db_size', type=int, default=1024)  # imle database size
    parser.add_argument('--imle_factor', type=float, default=0.)  # imle soft-sampling factor -- not used in the paper
    parser.add_argument('--imle_staleness', type=int, default=7)  # imle staleness, i.e., number of iterations to wait before considering the thresholds, tau_i
    parser.add_argument('--imle_batch', type=int, default=32)  # imle batch size used for sampling
    parser.add_argument('--subset_len', type=int, default=-1)  # subset length for training -- random subset of the dataset. -1 means full dataset
    parser.add_argument('--latent_dim', type=int, default=1024)  # latent code dimension
    parser.add_argument('--imle_perturb_coef', type=float, default=0.001)  # imle perturbation coefficient to avoid same latent codes
    parser.add_argument('--lpips_net', type=str, default='vgg')  # lpips network type
    parser.add_argument('--proj_dim', type=int, default=800)  # projection dimension for nearest neighbour search
    parser.add_argument('--proj_proportion', type=int, default=1)  # whether to use projection proportional to the lpips feature dimensions for nearest neighbour search
    parser.add_argument('--lpips_coef', type=float, default=1.0)  # lpips loss coefficient
    parser.add_argument('--l2_coef', type=float, default=0.1)  # l2 loss coefficient
    parser.add_argument('--clip_coef', type=float, default=0.1)
    parser.add_argument('--clip_temp', type=float, default=0.07)
    parser.add_argument('--dino_coef', type=float, default=1.0)  # l2 loss coefficient
    parser.add_argument('--force_factor', type=float, default=5)  # sampling factor for imle, i.e., force_factor * len(dataset)
    parser.add_argument('--change_coef', type=float, default=0.04)  # \gamma in the paper, rate of change of the thresholds, tau_i
    parser.add_argument('--change_threshold', type=float, default=1)  # starting threshold
    parser.add_argument('--n_mpl', type=int, default=8)  # mapping network layers
    parser.add_argument('--latent_lr', type=float, default=0.0001)  # learning rate for optimizing latent codes -- not used
    parser.add_argument('--latent_decay', type=float, default=0.0)  # learning rate decay for optimizing latent codes -- not used
    parser.add_argument('--latent_epoch', type=int, default=0)  # number of epochs for optimizing latent codes -- not used
    parser.add_argument('--reconstruct_iter_num', type=int, default=100000)  # number of iterations for reconstructing images using backtracking
    parser.add_argument('--imle_force_resample', type=int, default=5)  # number of iterations to wait before ignoringthe threshold and resample anyway
    parser.add_argument('--snoise_factor', type=int, default=8)  # spatial noise factor
    parser.add_argument('--max_hierarchy', type=int, default=256)  # maximum hierarchy level for spatial noise, i.e., 64 means up to 64x64 spatial noise but not higher resolution
    parser.add_argument('--load_strict', type=int, default=1)  # whether to load checkpoints strict
    parser.add_argument('--lpips_path', type=str, default='./lpips')  # path to lpips weights
    parser.add_argument('--image_size', type=int, default=256)  # image size of dataset -- possible to downsample the dataset
    parser.add_argument('--num_images_to_generate', type=int, default=100)
    parser.add_argument('--mode', type=str, default='train')  # mode of running, train, eval, reconstruct, generate
    
    parser.add_argument('--use_adaptive', default=False, type=lambda x: bool(x))  # whether to use adaptive imle

    parser.add_argument('--angle', type=float, default=0.0)  # angle to splatter
    parser.add_argument('--use_splatter', default=False, type=lambda x: bool(x))  # whether to use splatter
    
    parser.add_argument('--use_gaussian', default=False, type=lambda x: bool(x))  # whether to use splatter
    parser.add_argument('--gaussian_std', type=float, default=0.1)  # gaussian std
    # parser.add_argument('--mode', type=str, default='lpips', choices=['lpips', 'l2', 'combined']) # search type for nearest neighbour search

    parser.add_argument('--use_multi_res', default=True, type=lambda x: bool(x))  # whether to use nearest neighbour search
    parser.add_argument('--multi_res_scales', default='', type=str)  # extra multi-res dimension

    # parser.add_argument('--use_splatter_snoise', default=False, type=lambda x: bool(strtobool(x)))  # whether to use splatter snoise

    parser.add_argument('--use_snoise', default=False, type=lambda x: bool(x))  # whether to use spatial noise

    parser.add_argument('--search_type', type=str, default='lpips', choices=['lpips', 'l2', 'combined', 'vae']) # search type for nearest neighbour search
    parser.add_argument('--l2_search_downsample', type=float, default=0.125) # downsample factor for l2 search

    parser.add_argument('--use_angular_resample', default=False, type=lambda x: bool(x))  # whether to use spatial noise
    parser.add_argument('--use_eps_ignore', default=False, type=lambda x: bool(x))  # whether to use spatial noise
    # parser.add_argument('--use_eps_ignore_advanced', default=False, type=lambda x: bool(strtobool(x)))  # whether to use spatial noise
    parser.add_argument('--randomness_angular', type=float, default=0.0)  # whether to use splatter


    parser.add_argument('--eps_radius', type=float, default=0.1)  # angle to splatter
    parser.add_argument('--knn_ignore', type=int, default=5)  # whether to use spatial noise

    parser.add_argument('--max_sample_angle', type=float, default=180.0)  # max angle used for sampling
    parser.add_argument('--min_sample_angle', type=float, default=0.0)  # min angle used for sampling

    parser.add_argument('--wandb_name', type=str, default='AdaptiveIMLE')  # used for wandb
    parser.add_argument('--wandb_project', type=str, default='AdaptiveIMLE')  # used for wandb
    parser.add_argument('--use_wandb', type=int, default=1)
    parser.add_argument('--wandb_mode', type=str, default='online')
    parser.add_argument('--wandb_id', type=str, default=None)

    parser.add_argument('--use_comet', default=False, type=lambda x: bool(x))
    parser.add_argument('--comet_name', type=str, default='AdaptiveIMLE')  # used in comet.ml
    parser.add_argument('--comet_api_key', type=str, default='')  # comet.ml api key -- leave blank to disable comet.ml
    parser.add_argument('--comet_experiment_key', type=str, default='')

    parser.add_argument("--convnext_expansion", type=int, default=4, help="expansion factor for convnext")
    parser.add_argument("--use_se", default=False, type=lambda x: bool(x))  # whether to use se block
    parser.add_argument("--se_reduction", type=int, default=16, help="reduction factor for se block")
    parser.add_argument("--dropout_p", type=float, default=0.0, help="dropout rate for convnext block")


    # some metric args
    parser.add_argument("--space", choices=["z", "w"], help="space that PPL calculated with")
    parser.add_argument("--batch", type=int, default=16, help="batch size for the models")
    parser.add_argument("--n_sample", type=int, default=5000, help="number of the samples for calculating PPL",)
    parser.add_argument("--size", type=int, default=256, help="output image sizes of the generator")
    parser.add_argument("--eps", type=float, default=1e-4, help="epsilon for numerical stability")
    parser.add_argument("--ppl_snoise", type=int, default=0, help="whether to interpolate spatial noise in PPL")
    parser.add_argument("--sampling", default="end", choices=["end", "full"], help="set endpoint sampling method",)
    parser.add_argument("--step", type=float, default=0.1, help="step size for interpolation")
    parser.add_argument('--ppl_save_name', type=str, default='ppl')
    parser.add_argument("--fid_factor", type=int, default=5, help="number of the samples for calculating FID")
    parser.add_argument("--fid_freq", type=int, default=500, help="frequency of calculating fid")
    return parser

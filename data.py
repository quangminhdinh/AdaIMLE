import numpy as np
import pickle
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import clip
import json

from helpers.utils import get_world_size, crop_resize
from models import parse_layer_string
from torchvision.datasets import CIFAR10, STL10
from dataloaders import TextCLIPCondDataset


def set_up_data(H):
    if H.dataset in ['flowers102-t']:
        return set_up_data_wtext(H)
    return set_up_data_img(H)


def set_up_data_img(H):    
    blocks = parse_layer_string(H.dec_blocks)
    H.block_res = [s[0] for s in blocks]
    H.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))

    shift_loss = -127.5
    scale_loss = 1. / 127.5
    if H.dataset == 'imagenet32':
        trX, vaX, teX = imagenet32(H.data_root)
        H.image_size = 32
        H.image_channels = 3
        shift = -116.2373
        scale = 1. / 69.37404
    elif H.dataset in ['fewshot', 'fewshot512']:
        trX, vaX, teX = few_shot_image_folder(H.data_root, H.image_size)
        H.image_channels = 3
        shift = -116.2373
        scale = 1. / 69.37404
    elif H.dataset == 'flowers102-i':
        trX, vaX, teX = flowers102_img(H.image_size, H.data_root)
        H.image_channels = 3
        shift = -112.8666757481         # 71.93867001005759         93.6042881894389
        scale = 1. / 69.84780273        # 73.66214571500137         65.3031711042093
    elif H.dataset == 'imagenet64':
        trX, vaX, teX = imagenet64(H.data_root)
        H.image_size = 64
        H.image_channels = 3
        shift = -115.92961967
        scale = 1. / 69.37404
    elif H.dataset == 'ffhq_256':
        trX, vaX, teX = ffhq256(H.data_root)
        H.image_size = 256
        H.image_channels = 3
        shift = -112.8666757481
        scale = 1. / 69.84780273
    elif H.dataset == 'ffhq_1024':
        trX, vaX, teX = ffhq1024(H.data_root)
        H.image_size = 1024
        H.image_channels = 3
        shift = -0.4387
        scale = 1.0 / 0.2743
        shift_loss = -0.5
        scale_loss = 2.0
    elif H.dataset == 'cifar10':
        (trX, _), (vaX, _), (teX, _) = cifar10(H.data_root, one_hot=False)
        H.image_size = 32
        H.image_channels = 3
        shift = -120.63838
        scale = 1. / 64.16736
    elif H.dataset == "stl10":
        trX, vaX, teX = stl10(H.data_root)
        H.image_size = 64
        H.image_channels = 3
        shift = -0.5    
        scale = 1.0 / 0.5
    else:
        raise ValueError('unknown dataset: ', H.dataset)

    do_low_bit = H.dataset in ['ffhq_256']

    if H.test_eval:
        print('DOING TEST')
        eval_dataset = teX
    else:
        eval_dataset = vaX

    device = torch.device("cuda", torch.cuda.current_device())

    shift = torch.tensor([shift], device=device).view(1, 1, 1, 1)
    scale = torch.tensor([scale], device=device).view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss], device=device).view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss], device=device).view(1, 1, 1, 1)

    if H.dataset == 'ffhq_1024':
        train_data = ImageFolder(trX, transforms.ToTensor())
        valid_data = ImageFolder(eval_dataset, transforms.ToTensor())
        untranspose = True
    elif H.dataset == 'stl10':
        train_data = trX
        for data_train in DataLoader(train_data, batch_size=len(train_data)):
            ds = torch.tensor((data_train[0] + 1)/2 * 255, dtype=torch.uint8)
            train_data = TensorDataset(ds.permute(0, 2, 3, 1))
            break
        valid_data = train_data
        untranspose = False
    elif H.dataset not in ['fewshot', 'fewshot512']:
        train_data = TensorDataset(torch.as_tensor(trX))
        valid_data = TensorDataset(torch.as_tensor(eval_dataset))
        untranspose = False
    else:
        train_data = trX
        for data_train in DataLoader(train_data, batch_size=len(train_data)):
            ds = torch.tensor(data_train[0] * 255, dtype=torch.uint8)
            train_data = TensorDataset(ds.permute(0, 2, 3, 1))
            break
        valid_data = train_data
        untranspose = False
    
        
    H.global_batch_size = H.n_batch * get_world_size()
    H.total_iters = H.num_epochs * np.ceil(len(train_data) // H.global_batch_size)



    def preprocess_func(x):
        nonlocal shift
        nonlocal scale
        nonlocal shift_loss
        nonlocal scale_loss
        nonlocal do_low_bit
        nonlocal untranspose
        'takes in a data example and returns the preprocessed input'
        'as well as the input processed for the loss'
        if untranspose:
            x[0] = x[0].permute(0, 2, 3, 1)
        inp = x[0].to(device=device, non_blocking=True).float()
        inp.mul_(1./127.5).add_(-1)
        # out = inp.clone()
        # inp.add_(shift).mul_(scale)
        # if do_low_bit:
        #     5 bits of precision
        #     out.mul_(1. / 8.).floor_().mul_(8.)
        # out.add_(shift_loss).mul_(scale_loss)
        return inp, inp

    return H, train_data, valid_data, preprocess_func


def set_up_data_wtext(H):
    H.use_text = True

    blocks = parse_layer_string(H.dec_blocks)
    H.block_res = [s[0] for s in blocks]
    H.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))

    shift_loss = -127.5
    scale_loss = 1. / 127.5
    if H.dataset == 'flowers102-t':
        train, valid = flowers102_text(H.image_size, H.data_root, H.use_clip_loss)
        H.image_channels = 3
        shift = -112.8666757481         # 71.93867001005759         93.6042881894389
        scale = 1. / 69.84780273        # 73.66214571500137         65.3031711042093
    else:
        raise ValueError('unknown dataset: ', H.dataset)

    do_low_bit = H.dataset in ['ffhq_256']

    device = torch.device("cuda", torch.cuda.current_device())

    shift = torch.tensor([shift], device=device).view(1, 1, 1, 1)
    scale = torch.tensor([scale], device=device).view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss], device=device).view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss], device=device).view(1, 1, 1, 1)

    train_data = TextCLIPCondDataset(train, H)
    valid_data = TextCLIPCondDataset(valid, H)
    untranspose = False
            
    H.global_batch_size = H.n_batch * get_world_size()
    H.total_iters = H.num_epochs * np.ceil(len(train_data) // H.global_batch_size)

    def preprocess_func(x):
        nonlocal shift
        nonlocal scale
        nonlocal shift_loss
        nonlocal scale_loss
        nonlocal do_low_bit
        nonlocal untranspose
        'takes in a data example and returns the preprocessed input'
        'as well as the input processed for the loss'
        if untranspose:
            x[0] = x[0].permute(0, 2, 3, 1)
        inp = x[0].to(device=device, non_blocking=True).float()
        inp.mul_(1./127.5).add_(-1)
        return inp, inp

    return H, train_data, valid_data, preprocess_func


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def flatten(outer):
    return [el for inner in outer for el in inner]


def unpickle_cifar10(file):
    fo = open(file, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    data = dict(zip([k.decode() for k in data.keys()], data.values()))
    return data


def few_shot_image_folder(data_root, image_size):
    transform_list = [
        transforms.Resize((int(image_size), int(image_size))),
        transforms.ToTensor(),
    ]
    trans = transforms.Compose(transform_list)
    train_data = ImageFolder(data_root, trans)
    return train_data, train_data, train_data


def imagenet32(data_root):
    trX = np.load(os.path.join(data_root, 'imagenet32-train.npy'), mmap_mode='r')
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-5000]]
    valid = trX[tr_va_split_indices[-5000:]]
    test = np.load(os.path.join(data_root, 'imagenet32-valid.npy'), mmap_mode='r')
    return train, valid, test


def imagenet64(data_root):
    trX = np.load(os.path.join(data_root, 'imagenet64-train.npy'), mmap_mode='r')
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-5000]]
    valid = trX[tr_va_split_indices[-5000:]]
    test = np.load(os.path.join(data_root, 'imagenet64-valid.npy'), mmap_mode='r')  # this is test.
    return train, valid, test


def ffhq1024(data_root):
    # we did not significantly tune hyperparameters on ffhq-1024, and so simply evaluate on the test set
    return os.path.join(data_root, 'ffhq1024/train'), os.path.join(data_root, 'ffhq1024/valid'), os.path.join(data_root, 'ffhq1024/valid')


def ffhq256(data_root):
    trX = np.load(os.path.join(data_root, 'ffhq-256.npy'), mmap_mode='r')
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-7000]]
    valid = trX[tr_va_split_indices[-7000:]]
    # we did not significantly tune hyperparameters on ffhq-256, and so simply evaluate on the test set
    return train, valid, valid

def stl10(data_root):

    dataset = STL10("./data_stl", split="unlabeled", transform=transforms.Compose([
                            transforms.Resize(64),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)
    
    return dataset, None, None


def flowers102_img(img_size, data_root):
    ds = load_dataset("efekankavalci/flowers102-captions", split="train")
    trX = []
    # for i in tqdm(range(len(ds)), desc="Preprocessing flowers102-i:"):
    p = f'{data_root}/img'
    save_f = not os.path.exists(p)
    if save_f:
        os.makedirs(data_root, exist_ok=True)
        os.makedirs(p, exist_ok=True)
    for i in tqdm(range(1000), desc="Preprocessing flowers102-i:"):
        trX.append(crop_resize(np.asarray(ds[i]["image"]), img_size))
        if save_f:
            out = Image.fromarray(trX[-1])
            out.save(os.path.join(p, f"{i}.jpg"))
    trX = np.stack(trX) # b, h, w, c
    test_num = trX.shape[0] // 10
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-test_num]]
    valid = trX[tr_va_split_indices[-test_num:]]
    return train, valid, valid


def flowers102_text(img_size, data_root, use_img_emb=False):
    ds = load_dataset("efekankavalci/flowers102-captions", split="train")
    trX = []
    raw_txt = []
    txts = []
    imgs = [] if use_img_emb else None

    device = torch.device("cuda", torch.cuda.current_device())
    model, preprocess = clip.load('ViT-B/32', device)
    
    p = f'{data_root}/img'
    save_f = not os.path.exists(p)
    if save_f:
        os.makedirs(data_root, exist_ok=True)
        os.makedirs(p, exist_ok=True)
    
    with torch.no_grad():
        for i in tqdm(range(1000), desc="Preprocessing flowers102-t:"):
            img_path = os.path.join(p, f"{i}.jpg")
            if save_f:
                raw_img = crop_resize(np.asarray(ds[i]["image"]), img_size)
                out = Image.fromarray(raw_img)
                out.save(img_path)
            else:
                raw_img = np.asarray(Image.open(img_path))
            trX.append(raw_img)
            
            if use_img_emb:
                img = Image.fromarray(raw_img)
                image_input = preprocess(img).unsqueeze(0).to(device)
                imgs.append(model.encode_image(image_input).cpu())

            raw_txt.append(ds[i]["text"])
            text_input = clip.tokenize(raw_txt[-1]).to(device)
            txts.append(model.encode_text(text_input).cpu())
    trX = np.stack(trX) # b, h, w, c
    txts = torch.cat(txts)
    assert txts.shape[0] == trX.shape[0]
    assert txts.shape[0] == len(raw_txt)

    np.save(f'{data_root}/raw_imgs.npy', trX, allow_pickle=True)
    torch.save(txts, f'{data_root}/txts.pt')
    with open(f'{data_root}/raw_txts.json', 'w') as fp:
        json.dump(raw_txt, fp, indent=4)
    
    test_num = trX.shape[0] // 10
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = {
        "raw_img": trX[tr_va_split_indices[:-test_num]], 
        "raw_text": [raw_txt[i] for i in tr_va_split_indices[:-test_num]],
        "text": txts[tr_va_split_indices[:-test_num]],
    }
    valid = {
        "raw_img": trX[tr_va_split_indices[-test_num:]], 
        "raw_text": [raw_txt[i] for i in tr_va_split_indices[-test_num:]],
        "text": txts[tr_va_split_indices[-test_num:]],
    }
    if use_img_emb:
        imgs = torch.cat(imgs)
        assert txts.shape[0] == imgs.shape[0]
        torch.save(imgs, f'{data_root}/imgs.pt')
        train["img"] = imgs[tr_va_split_indices[:-test_num]]
        valid["img"] = imgs[tr_va_split_indices[-test_num:]]
    return train, valid


def cifar10(data_root, one_hot=True):
    tr_data = [unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'data_batch_%d' % i)) for i in range(1, 6)]
    trX = np.vstack([data['data'] for data in tr_data])
    trY = np.asarray(flatten([data['labels'] for data in tr_data]))
    te_data = unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'test_batch'))
    teX = np.asarray(te_data['data'])
    teY = np.asarray(te_data['labels'])
    trX = trX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    teX = teX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    trX, vaX, trY, vaY = train_test_split(trX, trY, test_size=5000, random_state=11172018)
    if one_hot:
        trY = np.eye(10, dtype=np.float32)[trY]
        vaY = np.eye(10, dtype=np.float32)[vaY]
        teY = np.eye(10, dtype=np.float32)[teY]
    else:
        trY = np.reshape(trY, [-1, 1])
        vaY = np.reshape(vaY, [-1, 1])
        teY = np.reshape(teY, [-1, 1])
    return (trX, trY), (vaX, vaY), (teX, teY)

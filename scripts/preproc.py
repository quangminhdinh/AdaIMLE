import numpy as np # type: ignore
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import clip
import json


def crop_resize(img, size):
    h, w, _ = img.shape
    if h > w:
        gap = h - w
        side = gap // 2
        img_t = img[side : side + w, ...]
    elif w > h:
        gap = w - h
        side = gap // 2
        img_t = img[:, side : side + h, :]
    else:
        img_t = img
    assert img_t.shape[0] == img_t.shape[1]
    im = Image.fromarray(img_t)
    im2 = im.resize((size, size))
    return np.asarray(im2)

img_size = 256
data_root = "/project/rrg-keli/qmd/datasets/imagenet"
# cache_dir = "/scratch/qmd/hf_cache"
use_img_emb = True

ds = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation")
trX = []
raw_txt = []
txts = []
imgs = [] if use_img_emb else None

device = torch.device("cuda")
model, preprocess = clip.load('ViT-B/32', device)

p = f'{data_root}/img'
small = f'{data_root}/small'
save_f = not os.path.exists(p)
if save_f:
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(p, exist_ok=True)
os.makedirs(small, exist_ok=True)

with torch.no_grad():
    for i in tqdm(range(len(ds)), desc="Preprocessing flowers102-t:"):
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

        raw_txt.append(ds[i]["caption_enriched"])
        text_input = clip.tokenize(raw_txt[-1]).to(device)
        txts.append(model.encode_text(text_input).cpu())
trX = np.stack(trX) # b, h, w, c
txts = torch.cat(txts)
assert txts.shape[0] == trX.shape[0]
assert txts.shape[0] == len(raw_txt)

np.save(f'{data_root}/raw_imgs.npy', trX, allow_pickle=True)
np.save(f'{small}/raw_imgs.npy', trX[:10000], allow_pickle=True)

torch.save(txts, f'{data_root}/txts.pt')
torch.save(txts[:10000], f'{small}/txts.pt')

with open(f'{data_root}/raw_txts.json', 'w') as fp:
    json.dump(raw_txt, fp, indent=4)
with open(f'{small}/raw_txts.json', 'w') as fp:
    json.dump(raw_txt[:10000], fp, indent=4)

if use_img_emb:
    imgs = torch.cat(imgs)
    assert txts.shape[0] == imgs.shape[0]
    torch.save(imgs, f'{data_root}/imgs.pt')
    torch.save(imgs[:10000], f'{small}/imgs.pt')
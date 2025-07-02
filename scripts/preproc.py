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
data_root = "/scratch/qmd/datasets/flowers_all"
use_img_emb = True

ds = load_dataset("efekankavalci/flowers102-captions", split="train")
trX = []
raw_txt = []
txts = []
imgs = [] if use_img_emb else None

device = torch.device("cuda")
model, preprocess = clip.load('ViT-B/32', device)

p = f'{data_root}/img'
save_f = not os.path.exists(p)
if save_f:
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(p, exist_ok=True)

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

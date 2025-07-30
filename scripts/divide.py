import numpy as np # type: ignore
import os
import torch
from PIL import Image
import json

data_root = "/scratch/qmd/datasets/celeba"

# txts = torch.load(f'{data_root}/txts.pt', map_location='cpu', weights_only=True)
# trX = np.load(f'{data_root}/raw_imgs.npy', allow_pickle=True)
with open(f'{data_root}/raw_txts.json', 'r') as fp:
  raw_t = json.load(fp)
# imgs = torch.load(f'{data_root}/imgs.pt', map_location='cpu', weights_only=True)

# np.save(f'{data_root}/small/raw_imgs.npy', trX[:10000], allow_pickle=True)
# torch.save(txts[:10000], f'{data_root}/small/txts.pt')
# torch.save(imgs[:10000], f'{data_root}/small/imgs.pt')
with open(f'{data_root}/small/raw_txts.json', 'w') as fp:
  json.dump(raw_t[:10000], fp, indent=4)

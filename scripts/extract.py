import numpy as np # type: ignore
from tqdm import tqdm
from PIL import Image

data_root = "/scratch/qmd/datasets/celeba"

trX = np.load(f'{data_root}/raw_imgs.npy', allow_pickle=True)
trX = trX[:5000]
# img

for i in tqdm(range(5000)):
  out = Image.fromarray(trX[i])
  out.save(f'{data_root}/img/{i}.png')




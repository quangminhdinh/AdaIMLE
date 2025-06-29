import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from fast_pytorch_kmeans import KMeans

from helpers.utils import is_main_process


class TextCLIPCondDataset(Dataset):
  
  def __init__(self, data, H, device):    
    super().__init__()
    
    print(f"\n{self.__class__.__name__}'s configurations.")

    self.trX = torch.as_tensor(data["raw_img"])
    self.txt_list = data["raw_text"]
    self.txt_clip = data["text"]
    self.img_clip = data["img"] if "img" in data else None

    if H.subset_len != -1:
      self.trX = self.trX[:H.subset_len, ...]
      self.txt_list = self.txt_list[:H.subset_len]
      self.txt_clip = self.txt_clip[:H.subset_len, ...]
      if self.img_clip is not None:
        self.img_clip = self.img_clip[:H.subset_len, ...]
    self.latent = None
    
    if H.random_proj_sz > 0:
      if(is_main_process()):
        if H.normalize_random_proj:
          path = f'{H.data_root}/proj{H.random_proj_sz}_norm.pt'
        else:
          path = f'{H.data_root}/proj{H.random_proj_sz}.pt'
        if os.path.exists(path):
          proj = torch.load(path, map_location='cpu', weights_only=True)
        else:
          proj = torch.randn(512, H.random_proj_sz, device="cpu", dtype=torch.half)
          if H.normalize_random_proj:
            proj = F.normalize(proj, p=2, dim=1)
          torch.save(proj, path)
        proj = proj.to(device)
      else:
        proj = torch.empty(512, H.random_proj_sz, device=device)
      
      torch.distributed.barrier()
      torch.distributed.broadcast(proj, src=0)
      torch.distributed.barrier()
      self.txt_clip = torch.mm(self.txt_clip, proj.cpu())
      self.rand_proj = proj.cpu()
    else:
      self.rand_proj = None

    if H.n_clusters > 0:
      self.kmeans = KMeans(n_clusters=H.n_clusters, mode='euclidean', verbose=1)
      labels = self.kmeans.fit_predict(self.txt_clip)
      self.txt_clip = self.kmeans.centroids[labels]
    else:
      self.kmeans = None
    if H.text_unit_norm:
      self.txt_clip = F.normalize(self.txt_clip, p=2, dim=1)
  
  def update_latent(self, latent):
    self.latent = latent

  def __len__(self):
    return len(self.trX)
  
  def __getitem__(self, idx):
    ret = {
      "raw_img": self.trX[idx], 
      "raw_text": self.txt_list[idx],
      "text": self.txt_clip[idx],
      "latent": self.latent[idx]
    }
    if self.img_clip is not None:
      ret["img"] = self.img_clip[idx]
    return ret


def text_clip_cond_collate(batch):
  bs = len(batch)
  
  ret = {
    "raw_img": torch.stack([batch[i]["raw_img"] for i in range(bs)]),
    "raw_text": [batch[i]["raw_text"] for i in range(bs)],
    "text": torch.stack([batch[i]["text"] for i in range(bs)]),
    "latent": torch.stack([batch[i]["latent"] for i in range(bs)]),
  }
  
  if "img" in batch[0]:
    ret["img"] = torch.stack([batch[i]["img"] for i in range(bs)])
  
  return ret

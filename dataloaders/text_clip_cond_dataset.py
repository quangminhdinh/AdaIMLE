import torch
from torch.utils.data import Dataset


class TextCLIPCondDataset(Dataset):
  
  def __init__(self, data, H):    
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

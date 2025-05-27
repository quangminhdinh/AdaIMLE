import torch

from .text_clip_cond_dataset import TextCLIPCondDataset, text_clip_cond_collate


class ZippedDataset(torch.utils.data.Dataset):

    def __init__(self, *datasets):
        assert all(len(datasets[0]) == len(dataset) for dataset in datasets)
        self.datasets = datasets

    def __getitem__(self, index):
        # print(index, [len(x) for x in self.datasets])
        return tuple(dataset[index] for dataset in self.datasets), index

    def __len__(self):
        return len(self.datasets[0])

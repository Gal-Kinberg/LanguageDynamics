import torch
from torch.utils.data import Dataset


class DyckDataset(Dataset):
    def __init__(self, data_blocks):
        self.data = torch.tensor(data_blocks, dtype=torch.long)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return x, y

class DyckAutoencoderDataset(Dataset):
    def __init__(self, data_blocks):
        self.data = torch.tensor(data_blocks, dtype=torch.long)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.data[idx]
        return x, y
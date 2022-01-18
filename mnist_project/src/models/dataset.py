import torch
from torch.utils.data import Dataset


class CorruptMNISTDataset(Dataset):
    def __init__(self, file_path: str, train=True):
        file = torch.load(file_path)
        if train:
            self.x = file["train_x"]
            self.Y = file["train_Y"]
        else:
            self.x = file["test_x"]
            self.Y = file["test_Y"]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.x[idx], self.Y[idx]

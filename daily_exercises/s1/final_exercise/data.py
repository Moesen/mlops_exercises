import torch
import numpy as np
import os
from torch.utils.data import Dataset

class CorruptedMNISTDataset(Dataset):
    def __init__(self, train=True):
        corrupt_path = "../../../data/corruptmnist"

        extension = "train" if train else "test"
        files = [np.load(os.path.join(corrupt_path, x) )
            for x in os.listdir(corrupt_path) if extension in x]
        
        self.images = torch.from_numpy(
            np.concatenate([x["images"] for x in files], axis=0)
            ).float()
        self.labels = torch.from_numpy(
            np.concatenate([x["labels"] for x in files], axis=0)
            ).long()
        
        self.img_dim = (28, 28)
        self.flatten_dim = (784)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        return img, label



def mnist(train=True):
    corrupt_path = "../../../data/corruptmnist"

    extension = "train" if train else "test"
    files = [np.load(os.path.join(corrupt_path, x) )
            for x in os.listdir(corrupt_path) if extension in x]
    
    images = torch.from_numpy(
        np.concatenate([x["images"] for x in files], axis=0)
    )
    labels = torch.from_numpy(
        np.concatenate([x["labels"] for x in files], axis=0)
    )

    return images, labels




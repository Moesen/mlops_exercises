"""
LFW dataloading
"""
import argparse
import time
import os

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import json


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        images = []
        for folder in os.listdir(path_to_folder):
            name = folder
            img_paths = os.listdir(os.path.join(path_to_folder, name))
            images.extend([
                {"name": name, "img_path": os.path.join(path_to_folder, name, img_path)}
                for img_path in img_paths
            ])

        self.data = images
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.data[index]["img_path"])
        transform = self.transform(img)
        return transform

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='', type=str)
    parser.add_argument('-num_workers', default=None, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
                            num_workers=args.num_workers)
    
    if args.visualize_batch:
        imgs = next(iter(dataloader))
        grid = make_grid(imgs)
        img = F.to_pil_image(grid)
        plt.imshow(np.asarray(img))
        
        
    if args.get_timing:
        times = []
        for worker_num in range(1, 13):
            print(worker_num)
            dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
                                num_workers=worker_num)
            # lets do so repetitions
            res = [ ]
            for _ in range(5):
                start = time.time()
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx > 100:
                        break
                end = time.time()

                res.append(end - start)
                
            res = np.array(res)
            times.append(res)
        calculated_times = [{"mean": np.mean(arr), "std": np.std(arr), "processors": i} for i, arr in enumerate(times)]
        with open("times.json", "w") as f:
            f.write(json.dumps(calculated_times))

import os
import pandas as pd
from torchvision.io import read_image
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths = sorted(os.listdir(img_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_paths[idx])
        image = read_image(img_path)
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])
        mask = read_image(mask_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask, img_path, mask_path
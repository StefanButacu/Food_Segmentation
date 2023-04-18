import os

import torch
from PIL import Image

import numpy as np
from torch.utils.data import Dataset


class FoodDataset(Dataset):
    def __init__(self,image_paths, seg_paths,transform = None):
        self.transforms = transform
        self.images,self.masks = [],[]
        for i in image_paths:
            imgs = os.listdir(i)
            self.images.extend([i+'/'+img for img in imgs])
        for i in seg_paths:
            masks = os.listdir(i)
            self.masks.extend([i+'/'+mask for mask in masks])
        self.images = self.images
        self.masks = self.masks

    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        img = np.array(Image.open(self.images[index]))
        mask = np.array(Image.open(self.masks[index]))
        mask = [[[category, 0, 0] for category in row] for row in mask]
        mask = np.array(mask)

        if self.transforms is not None:
            aug = self.transforms(image=img,mask=mask)
            img = aug['image']
            mask = aug['mask']
            mask = torch.max(mask,dim=2)[0]
        return img,mask
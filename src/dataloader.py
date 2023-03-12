import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
import imageio

class IF300Dataset(Dataset):
    """Custom Dataset for loading cropped IF300 images"""
    
    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path, index_col=0, header=None)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df[1].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]

custom_transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    transforms.Resize((32, 32))])

augment_transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    transforms.Resize((64, 64)),
                    transforms.Pad(4),
                    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.2)),
                    transforms.Resize((32, 32))])

train_dataset = IF300Dataset(csv_path='../data/train/train_labels.csv',
                              img_dir='../data/train/images',
                              transform=custom_transform)

val_dataset = IF300Dataset(csv_path='../data/val/val_labels.csv',
                              img_dir='../data/val/images',
                              transform=custom_transform)

test_dataset = IF300Dataset(csv_path='../data/test/test_labels.csv',
                             img_dir='../data/test/images',
                             transform=custom_transform)

BATCH_SIZE=64


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)

val_loader = DataLoader(dataset=val_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=4)
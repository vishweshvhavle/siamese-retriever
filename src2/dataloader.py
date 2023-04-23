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

class IF5000Dataset(Dataset):
    """Custom Dataset for loading cropped IF5000 images"""

    def __init__(self, csv_path, img_dir, transform=None, remove_ratio=0.5):
        df = pd.read_csv(csv_path, index_col=0, header=None)
        self.img_dir = img_dir
        self.img_paths = df.index.values
        self.labels = df.values.flatten()
        self.transform = transform
        self.remove_ratio = remove_ratio
        
        # Shuffle the indices
        self.shuffle_indices = np.random.permutation(len(self.img_paths))
        self.img_paths = self.img_paths[self.shuffle_indices]
        self.labels = self.labels[self.shuffle_indices]
        
        # Compute the number of samples to keep in each batch
        self.batch_size = None
        self.num_batches = None

    def __len__(self):
        return len(self.img_paths) // 2

    def __getitem__(self, idx):
        idx *= 2
        img_path_1 = os.path.join(self.img_dir, self.img_paths[idx])
        img_path_2 = os.path.join(self.img_dir, self.img_paths[idx+1])
        img_1 = Image.open(img_path_1).convert('RGB')
        img_2 = Image.open(img_path_2).convert('RGB')
        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        label = int(self.labels[idx] == self.labels[idx+1])
        
        # Remove some samples with label 0
        if label == 0:
            remove = np.random.choice([True, False], p=[1 - 0.25, 0.25])
            if remove:
                return self.__getitem__(np.random.randint(self.__len__()))
        
        return img_1, img_2, label


custom_transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    transforms.Resize((100, 100))])


train_dataset = IF5000Dataset(csv_path='../data/train/train_labels.csv',
                              img_dir='../data/train/images',
                              transform=custom_transform)

val_dataset = IF5000Dataset(csv_path='../data/val/val_labels.csv',
                              img_dir='../data/val/images',
                              transform=custom_transform)

test_dataset = IF5000Dataset(csv_path='../data/test/test_labels.csv',
                             img_dir='../data/test/images',
                             transform=custom_transform)

BATCH_SIZE=32


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
                         shuffle=True,
                         num_workers=4)
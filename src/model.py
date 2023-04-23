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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64*14*14, 22)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*14*14)
        x = self.fc(x)
        return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.vgg = torchvision.models.vgg16(pretrained=True)
#         for param in self.vgg.parameters():
#             param.requires_grad = False
#         self.fc = nn.Sequential(nn.Linear(1000, 100),
#                                  nn.ReLU(),
#                                  nn.Linear(100, 8))

#     def forward(self, x):
#         x = self.vgg(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
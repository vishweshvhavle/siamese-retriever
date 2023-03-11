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

class Identity2(nn.Module):
    def __init__(self):
        super(Identity1, self).__init__()


    def forward(self, x):
        return x

net2 = torchvision.models.resnet18(pretrained=True)
for param in net2.parameters():
    param.requires_grad = False
net2.fc = nn.Sequential(nn.Linear(512, 100),
                        nn.ReLU(),
                        nn.Linear(100, 10))
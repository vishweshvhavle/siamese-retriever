import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
import imageio

# BATCH_SIZE = 32
# IMAGE_SIZE = 100 X 100

class SiameseResNet(nn.Module):
    def __init__(self):
        super(SiameseResNet, self).__init__()
        
        self.resnet = models.resnet50(pretrained=False)
        out_features = list(self.resnet.modules())[-1].out_features
        self.fc1 = nn.Linear(2*1000, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 2)
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward_once(self, x):
        # Pass the input tensor through the pre-trained ResNet50 model
        features = self.resnet(x)
        out = features.view(features.size(0), -1)
        return out

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        
        output = output1 * output2
        output = self.cls_head(output)
        return output

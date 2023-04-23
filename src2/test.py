import wandb
import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
import imageio
from tqdm import tqdm

from dataloader import *
from model1 import SiameseResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# load the saved model
model = SiameseResNet()
model.load_state_dict(torch.load('./if5000_model1.pth'))
model.eval()
model.to(device)

# define loss function
criterion = nn.BCEWithLogitsLoss()

# evaluate the model on the test set
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for data in tqdm(test_loader):
        img0, img1 , labels = data
        img0, img1 , labels = img0.to(device), img1.to(device) , labels.to(device)
        outputs = model(img0,img1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy:', 100 * correct / total)
print('Test Loss:', test_loss / len(test_loader))
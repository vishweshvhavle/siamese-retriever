import pandas as pd
import numpy as np
import os
import shutil

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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dataloader import *
from model import *

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = './if20K_model1.pth'
model = Net()
model.load_state_dict(torch.load(PATH))
model.to(device)

if not os.path.exists('structured_data'):
    os.mkdir('structured_data')

for i in range(20):
    if not os.path.exists(f'structured_data/{i}'):
        os.mkdir(f'structured_data/{i}')

filenames_list = []

with torch.no_grad():
    for data in test_loader:
        images, labels, filenames = data

        filenames_list.append(filenames)
        images = images.to(device)

        outputs = model(images)
        outputs_ = np.array(outputs.cpu())

        _, predicted = torch.max(outputs.data, 1)


        predicted = predicted.tolist()
        # print(filenames_list)
        for i in range(len(predicted)):
            # print(filenames_list[i])
            # print(predicted[i])
            class_dir = 'structured_data/' + str(predicted[i])
            src_dir = '../data/test/images/' + filenames_list[0][i]
            shutil.copy(src_dir, class_dir)
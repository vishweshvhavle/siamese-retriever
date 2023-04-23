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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dataloader import *
from model import *

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = './if20K_model1.pth'
model = Net()
model.load_state_dict(torch.load(PATH))
model.to(device)

transform = transforms.ToPILImage()

y_true = []
y_pred = []

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        for i in range(len(labels)):
            if labels[i] == 10:
                labels[i] = 0
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        labels_ = np.array(labels.cpu())
        outputs_ = np.array(outputs.cpu())

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        labels = labels.tolist()
        predicted = predicted.tolist()
        y_true += labels
        y_pred += predicted

print(f'Accuracy of the network on the 3121 test images: ', 100 * correct // total)
print(f'F1 Score of the network on the 3121 test images: ', f1_score(y_true, y_pred, average='weighted'))

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        for i in range(len(labels)):
            if labels[i] == 10:
                labels[i] = 0
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / (float(total_pred[classname])+0.001)
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
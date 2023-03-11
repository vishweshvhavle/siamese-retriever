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
from model2 import *

classes = ['0','1','2','3','4','5','6','7','8','9']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dataiter = iter(test_loader)
# images, labels = next(dataiter)
# images, labels = images.to(device), labels.to(device)

# print('GroundTruth: ', ' '.join(f'{classes[labels[j]-1]:5s}' for j in range(40)))

PATH = './svhn_net3.pth'
net = net2
net.load_state_dict(torch.load(PATH))
net.to(device)

transform = transforms.ToPILImage()

# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join(f'{classes[predicted[j]-1]:5s}' for j in range(40)))

# for j in range(40):
#     if classes[labels[j]-1] != classes[predicted[j]-1]:
#         img = transform(images[j])
#         img.show()
#         print('True: ',classes[labels[j]-1])
#         print('Prediction: ',classes[predicted[j]-1])

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
        outputs = net(images)
        labels_ = np.array(labels.cpu())
        outputs_ = np.array(outputs.cpu())

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        labels = labels.tolist()
        predicted = predicted.tolist()
        y_true += labels
        y_pred += predicted

print(f'Accuracy of the network on the 14651 test images: ', 100 * correct // total)
print(f'F1 Score of the network on the 14651 test images: ', f1_score(y_true, y_pred, average='weighted'))

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        for i in range(len(labels)):
            if labels[i] == 10:
                labels[i] = 0
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
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
from model import SiameseResNet

classes = ('0','1','2','3','4','5','6','7')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

num_epochs = 3
model = SiameseResNet()
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

counter = []
loss_history = [] 
iteration_number= 0

# # start a new wandb run to track this script
# wandb.init(
#     entity = "cse-344", 
#     project = "final_review",
#     name = "net1"
# )

# wandb.config={
#     "learning_rate": 0.02,
#     "architecture": "CNN",
#     "dataset": "IF5000",
#     "epochs": 20,
#     "batch_size": 64
#     }

# wandb.watch(model)

for epoch in range(num_epochs):
    print("Epoch: ", epoch)
    batch_loss_train = 0
    batch_loss_val = 0
    iteration_number = 0

    with tqdm(train_loader, unit="batch") as t:
        for i, data in enumerate(t):
            img0, img1 , labels = data 
            img0, img1 , labels = img0.to(device), img1.to(device) , labels.to(device)
            optimizer.zero_grad()
            outputs = model(img0,img1)
            # print(f"Min target: {labels.min().item()}, Max target: {labels.max().item()}")
            labels = labels.view(-1, 1)
            labels = labels.to(torch.float32)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.item())
                t.set_postfix(loss=loss.item())

            batch_loss_train += loss.item()
            if(i == 15):
            	break
    
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for data in val_loader:
            img0, img1 , labels = data
            img0, img1 , labels = img0.to(device), img1.to(device) , labels.to(device)
            outputs = model(img0,img1)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            labels = labels.view(-1, 1)
            labels = labels.to(torch.float32)

            loss_val = criterion(outputs, labels)
            batch_loss_val += loss_val.item()

    t.set_postfix(
        train_loss=batch_loss_train / len(train_loader),
        val_loss=batch_loss_val / len(val_loader),
        val_acc=correct_val / total_val,
    )

    
    accuracy = (100 * correct_val / total_val)
    print('Validation Accuracy', accuracy)
    
    # wandb.log(
    #     {"Accuracy" : accuracy , "Train_loss":batch_loss_train, "Val_loss": batch_loss_val}
    #     )

PATH = './if5000_model1.pth'
torch.save(model.state_dict(), PATH)

# artifact = wandb.Artifact('model', type='net')
# artifact.add_file(PATH)
# wandb.log_artifact(artifact)
# wandb.finish()
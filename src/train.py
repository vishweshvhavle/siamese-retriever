import wandb
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

from dataloader import *
from model import *

classes = ('0','1','2','3','4','5','6','7')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

num_epochs = 20
model = Net()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# start a new wandb run to track this script
wandb.init(
    entity = "cse-344", 
    project = "mid_review",
    name = "net1"
)

wandb.config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "IF5000",
    "epochs": 20,
    "batch_size": 64
    }

wandb.watch(model)

total_loss_train = []
total_loss_validation = []

for epoch in range(num_epochs):  # loop over the dataset multiple times

    batch_loss_train = 0
    batch_loss_val = 0
    num_batches = 0
    running_loss = 0.0

    print('Epoch: ', epoch + 1)

    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = loss.item()
        batch_loss_train += loss.item()

    num_batches+=1
    batch_loss_train/=num_batches
    print("Training Loss: ",batch_loss_train)

    total_loss_train.append(batch_loss_train)

    # Valdiation Loop
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        cnt = 0
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the modelwork
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=labels.cpu().detach().numpy(), preds=predicted.cpu().detach().numpy(),
                        class_names=classes)})

            # Compute loss 
            total += labels.size(0)
            cnt+=1
            loss_val = criterion(outputs, labels)
            batch_loss_val += loss_val.item()
  
    acc = 100 * correct / total
    total_loss_validation += [batch_loss_val/cnt]
    # ✨ W&B: Log accuracy across training epochs, to visualize in the UI
    wandb.log(
        {"Accuracy" : acc , "Train_loss":batch_loss_train, "Val_loss": batch_loss_val}
        )


    print("Validation Loss: ",batch_loss_val/cnt)
    print("Validation Accuracy: ", acc)

print('Finished Training')

PATH = './if5000_model1.pth'
torch.save(model.state_dict(), PATH)
artifact = wandb.Artifact('model', type='net')
artifact.add_file(PATH)
wandb.log_artifact(artifact)
wandb.finish()

plt.plot(total_loss_validation)
plt.title("Validation loss")
plt.show()

plt.plot(total_loss_train)
plt.title("Train Loss")
plt.show()
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
import streamlit as st
import PIL
from PIL import Image
import cv2
import random

custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((32, 32))
])

classes = ['0', '1', '2', '3', '4']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = './if300_model1.pth'
model.load_state_dict(torch.load(PATH))
model.to(device)

try:
    img_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if img_file is not None:
        img = Image.open(img_file)
        img = img.resize((250,250))
        st.image(img)

        with torch.no_grad():
            img_transformed = custom_transform(img)
            img_batch = torch.stack([img_transformed, img_transformed]).to(device)
            output = model(img_batch)
            _, predicted = torch.max(output.data, 1)

            predicted = predicted.tolist()
            predicted_class = str(predicted[0])

            st.write(f"Predicted class: {predicted_class}")
            labelPic = predicted[0]
            class_dir = 'structured_data/' + predicted_class

            l=[]
            for i in os.listdir(class_dir):
                l.append(i)

            random.shuffle(l)
            c=0
            for i in l:
                if(c==10):
                    break
                else:
                    x=Image.open(class_dir+'/'+i)
                    x = x.resize((250,250))
                    st.image(x)
                    c+=1

    else:
        st.write("Upload an image to get started!")
except Exception as e:
    st.write(f"Error: {e}")

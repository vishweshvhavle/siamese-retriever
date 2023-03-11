import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
import imageio

main_dir='../datasets'
mat_file='../datasets/train_32x32.mat'
    
if not os.path.exists(main_dir):
    os.mkdir(main_dir)

data = sio.loadmat(mat_file)

x = np.transpose(data['X'], (3, 0, 1, 2))
y = data['y'].flatten()

# Train Set Size: 73257 Digits
# Custom Train Set Size: 51281 Digits
# Custom Test Set Size: 14651 Digits
# Custom Val Set Size: 7325 Digits

label='test'    
sub_dir = os.path.join(main_dir, label)
if not os.path.exists(sub_dir):
    os.mkdir(sub_dir)

with open(os.path.join(main_dir, '%s_labels.csv' % label), 'w') as out_f:
    for i in range(14651):
        img = x[i]
        file_path = os.path.join(sub_dir, str(i) + '.png')
        imageio.imwrite(os.path.join(file_path), img)

        out_f.write("%d.png,%d\n" % (i, y[i]))

label='val'    
sub_dir = os.path.join(main_dir, label)
if not os.path.exists(sub_dir):
    os.mkdir(sub_dir)

with open(os.path.join(main_dir, '%s_labels.csv' % label), 'w') as out_f:
    for i in range(14652, 21976):
        img = x[i]
        file_path = os.path.join(sub_dir, str(i) + '.png')
        imageio.imwrite(os.path.join(file_path), img)

        out_f.write("%d.png,%d\n" % (i, y[i]))

label='train'    
sub_dir = os.path.join(main_dir, label)
if not os.path.exists(sub_dir):
    os.mkdir(sub_dir)

with open(os.path.join(main_dir, '%s_labels.csv' % label), 'w') as out_f:
    for i in range(21977, 73256):
        img = x[i]
        file_path = os.path.join(sub_dir, str(i) + '.png')
        imageio.imwrite(os.path.join(file_path), img)

        out_f.write("%d.png,%d\n" % (i, y[i]))
import pandas as pd
import numpy as np
import os
import random

import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
import imageio
import csv

import glob
import shutil

main_dir='../data'

    
if not os.path.exists(main_dir):
    os.mkdir(main_dir)

# Dataset Set Size: 5640 Images
# Custom Train Set Size: 3948 Images
# Custom Test Set Size: 1128 Images
# Custom Val Set Size: 564 Images

with open('../data/original/labels.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    random_rows = random.sample(rows, len(rows))

    # Create Directories for Train Set
    label = 'train'    
    sub_dir = os.path.join(main_dir, label)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    train_dir = os.path.join(sub_dir, 'images')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    # Create Directories for Test Set
    label = 'test'    
    sub_dir = os.path.join(main_dir, label)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    test_dir = os.path.join(sub_dir, 'images')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # Create Directories for Val Set
    label = 'val'    
    sub_dir = os.path.join(main_dir, label)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    val_dir = os.path.join(sub_dir, 'images')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

    count = 0
    for row in random_rows:
        with open(os.path.join(main_dir, '%s/%s_labels.csv' % (label, label)), 'a') as out_f:
            filename = row[0]
            label_id = row[1]
            out_f.write("%s,%s\n" % (filename, label_id))
            dst_dir = os.path.join(main_dir, label, 'images')
            src_dir = os.path.join('../data/original/images', filename)
            shutil.copy(src_dir, dst_dir)

            count += 1
            if count == 1128:
                label = 'test'
            elif count == 1692:
                label = 'train'
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

org_dir = os.path.join('../data/','original')
if not os.path.exists(org_dir):
    os.mkdir(org_dir)

dst_dir = os.path.join('../data/original/', 'images')
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

data_dir = '../data_structured'
class_names = os.listdir(data_dir)
classes = {class_names[i]: i for i in range(len(class_names))}
class_names = list(classes.keys())
dic = {}

image_id = 1
for folder in glob.iglob(os.path.join('../data_structured', "*")):
	for file in glob.iglob(os.path.join(folder, "*.jpg")):
		label = folder[19:]
		os.rename(file, os.path.join(folder, label + '_image_' + str(image_id) + ".jpg") )
		image_id += 1

with open(os.path.join('../data/original/', 'labels.csv'), 'w') as out_f:
	for folder in glob.iglob(os.path.join('../data_structured', "*")):
		label = folder[19:]
		label_id = classes[label]

		for file in glob.iglob(os.path.join(folder, "*.jpg")):
			shutil.copy(file, dst_dir)
			out_f.write("%s,%d\n" % (file[len(folder)+1:], label_id))

			if dic.get(label) is not None:
			    dic[label] += 1
			else:
			    dic[label] = 1
count = []
for n in classes:
    if n in dic:
        count.append(dic[n])
    else:
        count.append(0)

plt.title("Dataset Distribution")
plt.ylabel("Count")
plt.xticks(range(len(classes)), classes)

plt.bar(range(len(classes)), count)
plt.show()

main_dir='../data'
if not os.path.exists(main_dir):
    os.mkdir(main_dir)

jpg_count = len([f for f in os.listdir(dst_dir) if f.endswith('.jpg')])
print(jpg_count)
train_size = int(0.7*jpg_count)
test_size = int(0.2*jpg_count)
val_size = int(0.1*jpg_count)

print("Train set size: ", train_size)
print("Test set size: ", test_size)
print("Validation set size: ", val_size)

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
            if count == val_size:
                label = 'test'
            elif count == val_size + test_size:
                label = 'train'
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

df = pd.read_csv('../data/train/train_labels.csv', header=None, index_col=0)
arr = df[1].to_numpy()
dic = {}
for n in arr:
	if dic.get(n) is not None:
	    dic[n] += 1
	else:
	    dic[n] = 1

train = dic

df = pd.read_csv('../data/val/val_labels.csv', header=None, index_col=0)
arr = df[1].to_numpy()
dic = {}
for n in arr:
	if dic.get(n) is not None:
	    dic[n] += 1
	else:
	    dic[n] = 1

val = dic

df = pd.read_csv('../data/test/test_labels.csv', header=None, index_col=0)
arr = df[1].to_numpy()
dic = {}
for n in arr:
	if dic.get(n) is not None:
	    dic[n] += 1
	else:
	    dic[n] = 1

test = dic

classes = {'biryani': 0, 'butter_chicken': 1, 'masala_dosa': 2, 'pani_puri': 3, 'vada_pav': 4}
class_names = list(classes.keys())

nums = [0, 1, 2, 3, 4]
count = []
for n in nums:
	count.append(train[n])

plt.title("Train")
plt.ylabel("Count")
plt.xticks(range(len(classes)), class_names)

plt.bar(range(len(classes)), count)
plt.show()

count = []
for n in nums:
	count.append(val[n])

plt.title("Validation")
plt.ylabel("Count")
plt.xticks(range(len(classes)), class_names)

plt.bar(range(len(classes)), count)
plt.show()

count = []
for n in nums:
	count.append(test[n])

plt.title("Test")
plt.ylabel("Count")
plt.xticks(range(len(classes)), class_names)

plt.bar(range(len(classes)), count)
plt.show()
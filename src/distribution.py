import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

df = pd.read_csv('../datasets/svhn/train_labels.csv', header=None, index_col=0)
arr = df[1].to_numpy()
dic = {}
for n in arr:
	if dic.get(n) is not None:
	    dic[n] += 1
	else:
	    dic[n] = 1

train = dic

df = pd.read_csv('../datasets/svhn/val_labels.csv', header=None, index_col=0)
arr = df[1].to_numpy()
dic = {}
for n in arr:
	if dic.get(n) is not None:
	    dic[n] += 1
	else:
	    dic[n] = 1

val = dic

print(train)
print(val)

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
count = []
for n in nums:
	count.append(train[n])

plt.title("Train")
plt.ylabel("Count")

plt.bar(range(len(nums)), count)
plt.show()

count = []
for n in nums:
	count.append(val[n])

plt.title("Validation")
plt.ylabel("Count")

plt.bar(range(len(nums)), count)
plt.show()
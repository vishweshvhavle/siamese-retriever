import pandas as pd
import numpy as np
import os, sys

import matplotlib.pyplot as plt
from PIL import Image
import glob
import shutil

dst_dir = os.path.join('../data/', 'images')
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

classes = {'biryani': 0, 'butter_chicken': 1, 'masala_dosa': 2, 'pani_puri': 3, 'vada_pav': 4}
class_names = list(classes.keys())
dic = {}
# image_id = 1
with open(os.path.join('../data/', 'labels.csv'), 'w') as out_f:
	for folder in glob.iglob(os.path.join('../imagedataset', "*")):
		label = folder[16:]
		label_id = classes[label]


		for file in glob.iglob(os.path.join(folder, "*.jpg")):
			# os.rename(file, os.path.join(folder, str(image_id) + ".jpg") )
			# image_id += 1

			shutil.copy(file, dst_dir)
			out_f.write("%s,%d\n" % (file[len(folder)+1:], label_id))

			if dic.get(label) is not None:
			    dic[label] += 1
			else:
			    dic[label] = 1

count = []
for n in classes:
	count.append(dic[n])

plt.title("Dataset Distribution")
plt.ylabel("Count")
plt.xticks(range(len(classes)), class_names)

plt.bar(range(len(classes)), count)
plt.show()
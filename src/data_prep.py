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

dic = {}
# image_id = 1
with open(os.path.join('../data/', 'labels.csv'), 'w') as out_f:
	for folder in glob.iglob(os.path.join('../imagedataset', "*")):
		label = folder[16:]

		for file in glob.iglob(os.path.join(folder, "*.jpg")):
			# os.rename(file, os.path.join(folder, str(image_id) + ".jpg") )
			# image_id += 1

			shutil.copy(file, dst_dir)
			out_f.write("%s,%s\n" % (file[len(folder)+1:], label))
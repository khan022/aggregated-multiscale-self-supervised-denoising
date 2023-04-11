import glob
import os
import sys
import re
import cv2
import matplotlib.pyplot as plt

import PIL.Image
import numpy as np
import random
import math


#### Creating noisy npy file for easy dataloading
get_dir = glob.glob('location of the noisy images')

noisy_images = list()
for path in get_dir:
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    noisy_images.append(img)
    
noisy_images = np.asarray(noisy_images)
print(noisy_images.shape)### Printing shape to check if there is any data mis-match

np.save('./noisy.npy', noisy_images)


#### Creating target npy file for easy dataloading
get_dir = glob.glob('location of the target images')

target_images = list()
for path in get_dir:
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    target_images.append(img)
    
target_images = np.asarray(target_images)
print(target_images.shape)### Printing shape to check if there is any data mis-match

np.save('./target.npy', target_images)
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


## Function for generating noise
def nod(x,fc):
    nx = np.random.normal(scale=fc/255, size=x.shape)
    ad = np.clip(x+nx,0,1)
    return np.float32(ad)

dat_dir = glob.glob('/location of the main images')


## Loop to generate noisy image
for path in dat_dir:
    p_name = path.split('\\')[-1]
    i_name = p_name.split('.')[0]
    
    img = cv2.imread(path)
    img = cv2.resize(img, (500, 500), interpolation = cv2.INTER_AREA)
    noisy_1 = nod(img/255, random.randint(40, 90))
    noisy_2 = nod(img/255, random.randint(50, 100))
    noisy_3 = nod(img/255, random.randint(30, 80))
    
    fi1 = './noisy/'+i_name+'_1.png'
    fi2 = './noisy/'+i_name+'_2.png'
    fi3 = './noisy/'+i_name+'_3.png'
    
    cv2.imwrite(fi1, noisy_1*255)
    cv2.imwrite(fi2, noisy_2*255)
    cv2.imwrite(fi3, noisy_3*255)
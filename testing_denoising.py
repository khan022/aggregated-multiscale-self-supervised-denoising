import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
#import keras
import numpy as np
import os
from matplotlib.pyplot import figure
#from keras.backend import tensorflow_backend
from tensorflow.keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from skimage.metrics import structural_similarity as ssim
import math
import cv2
import glob
import re
import tensorflow as tf
from skimage import segmentation, color
from skimage.io import imread
from skimage.future import graph
from matplotlib import pyplot as plt
from tensorflow.keras.utils import Sequence
import numpy as np 


#### Miscellaneous functions

def plot_sample(lr, sr):
    plt.figure(figsize=(8, 6))

    images = [lr, sr]
    titles = ['denoised', 'noisy']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        
def sub3(f1,f2,f3):
 
    fig, ax = plt.subplots(1,3,figsize=(15,15))
    plt.subplot(131)
    plt.imshow(f1)
    plt.title("input image")
    plt.subplot(132)
    plt.imshow(f2)
    plt.title("denoised")
    plt.subplot(133)
    plt.imshow(f3)
    plt.title("noisy")
    plt.show()
    
    
def adik(im):
    a= cv2.cvtColor(im, cv2.COLOR_BGR2RGB)/255
    a= cv2.resize(a, (256,256), interpolation=cv2.INTER_CUBIC)
    return a  

def nod(x,fc):
    nx= np.random.normal(scale=fc/255, size=x.shape)
    ad= np.clip(x+nx,0,1)
    return np.float32(ad)

def lsr(im,fc):
    d1=im.shape[1]
    d2=im.shape[2]
    d3=round(d1/fc)
    d4=round(d2/fc)
    w1=  (tf.image.resize(im, [d3, d4]))
    return  (tf.image.resize(w1, [d1, d2])) 

def ox(a):
    return np.expand_dims(a,axis=0)


#### Creating the model for testing

### Modules for testing
def rcat(x,f):
    y1= Conv2D(f//4, kernel_size=3, strides=1, padding='same')(x)
    x1= Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
    x1= Conv2D(16, kernel_size=1, strides=1, padding='same')(x1)
    x1= Conv2D(16, kernel_size=5, strides=1, padding='same')(x1)
    x1= Conv2D(16, kernel_size=7, strides=1, padding='same')(x1)
    x1= Conv2D(16, kernel_size=3, strides=1, padding='same')(x1)
    a1=actc(x1)
    a2=actc(y1)
    c1=concatenate([a1,a2])
    c2=  Conv2D(f//4, kernel_size=1, strides=1, padding='same')(c1)
    return add([c2,y1])

def rdn(x,f):
    y1 = Conv2D(f, kernel_size=3, strides=1, padding='same')(x)
    y2 = Conv2D(f, kernel_size=3, strides=1, padding='same')(y1)
    a1 = add([y1,y2])
    y3 = Conv2D(f, kernel_size=3, strides=1, padding='same')(a1)
    a2 = add([y3,a1])
    y4 = Conv2D(f, kernel_size=3, strides=1, padding='same')(a2)
    a3 = add([a1,a2,y4])
    c  = concatenate([a1,a2,a3])
    return  Conv2D(f//4, kernel_size=3, strides=1, padding='same')(c)

def runt(x,f):
    y1 = Conv2D(f, kernel_size=3, strides=1, padding='same')(x)
    y1 = Conv2D(f, kernel_size=3, strides=1, padding='same')(y1)
    y2 = Conv2D(f//2, kernel_size=3, strides=1, padding='same')(y1)
    y2 = Conv2D(f//2, kernel_size=5, strides=1, padding='same')(y2)
    y3 = Conv2D(f//4, kernel_size=3, strides=1, padding='same')(y2)
    y3 = Conv2D(f//4, kernel_size=5, strides=1, padding='same')(y3)
    y4 = Conv2D(f//8, kernel_size=3, strides=1, padding='same')(y3)
    y4 = Conv2D(f//8, kernel_size=5, strides=1, padding='same')(y4)
    y5 = Conv2D(f//8, kernel_size=1, strides=1, padding='same')(y4)
    c1 = concatenate([y5,y4])
    y6 = Conv2D(f//4, kernel_size=1, strides=1, padding='same')(c1)
    c2 = concatenate([y6,y3])
    y7 = Conv2D(f//2, kernel_size=1, strides=1, padding='same')(c2)
    c3 = concatenate([y7,y2])
    y8 = Conv2D(f, kernel_size=1, strides=1, padding='same')(c3)
    y8 = add([y8,y1])
    y9 = Conv2D(3, kernel_size=1, strides=1, padding='same')(y8)
    return y9

def den(img):
    x= Conv2D(64, kernel_size=5, strides=1, padding='same')(img)  
    x= Conv2D(64, kernel_size=3, strides=1, padding='same')(x)  
    x= Conv2D(64, kernel_size=1, strides=1, padding='same')(x)  
    f1=actc(x)
    f2=mdsr1(x,32)
    f3=rdn(x,128)
    inp=concatenate([f1,f2,f3,x]) 
    x=Conv2D(3, 3, dilation_rate=2, strides=1, padding='same')(inp) 
    return x

def actc(x):      
    x1 = Activation('relu')(x)
    x2 = Activation('sigmoid')(x)
    x3 = Lambda(lambda x: x[0]*x[1])([x2,x])
    x4 = Activation('softplus')(x)
    x4=Activation('tanh')(x4)
    x5 = Lambda(lambda x: x[0]*x[1])([x4,x])
    c1= Conv2D(7, kernel_size=3, strides=1, padding='same')(x1)  
    c2= Conv2D(7, kernel_size=5, strides=1, padding='same')(x3)  
    c3= Conv2D(7, kernel_size=7, strides=1, padding='same')(x5)  
    cx=concatenate([c1,c2,c3], axis = 3)
    y= Conv2D(3, kernel_size=3, strides=1, padding='same')(cx)
    return y

def r1(input_tensor, features ):
    x = Conv2D(features, 3, activation='relu', padding='same')(input_tensor)
    x = Conv2D(features, 3, padding='same')(x)
    return add([input_tensor, x])
def mdsr1(ix,f):
    x=Conv2D(f, kernel_size=3, strides=1, padding='same')(ix)
    
    x1=r1(x,f)
    x1=r1(x1,f)
    x2=r1(x,f)
    x2=r1(x2,f)
    x3=r1(x,f)
    x3=r1(x3,f)
    x=add([x1,x2,x3])
    x=concatenate([x,x1,x2,x3 ], axis = 3)
    x=Conv2D(3, kernel_size=3, strides=1, padding='same')(x)
    return x 


#### Main model
input_im = Input(shape=(None,None,3))

input_img=Conv2D(64, 3, dilation_rate=2, strides=1, padding='same')(input_im) 
input_img=Conv2D(32, 1, dilation_rate=4, strides=1, padding='same')(input_img)
input_img=Conv2D(16, 1, dilation_rate=16, strides=1, padding='same')(input_img)

x = concatenate([den(input_img ),runt(input_img ,16)])    
x = Conv2D(3, kernel_size=1, dilation_rate=8, strides=1, padding='same')(x)
x = concatenate([den(x),runt(x, 32)])
y = Conv2D(3, kernel_size=1, dilation_rate=8, strides=1, padding='same')(x)

model = Model(input_im, y)


### Loading the weight in the model

model.load_weights('./weights/noise_weight.h5')


### loading sample images for testing

v = np.load('sample images for testing')


### Getting a sample image from v

od = 12
tim = xs[od]
 
h1 = tim.shape[0]
w1 = tim.shape[1]
ua = 1
lx = cv2.resize(tim, (w1//ua, h1//ua))

#### Here 3 different noise level is implemented on the sample image
n1 = (ox(nod(lx,10)))
n2 = (ox(nod(lx,30)))
n3 = (ox(nod(lx,50)))

#### Output for those 3 different noise level images are generated
p1 = np.clip(cv2.resize((model.predict(n1)[0] ), (w1,h1),  interpolation=cv2.INTER_CUBIC),0,1)
p2 = np.clip(cv2.resize((model.predict(n2)[0] ), (w1,h1),  interpolation=cv2.INTER_CUBIC),0,1)
p3 = np.clip(cv2.resize((model.predict(n3)[0]), (w1,h1),  interpolation=cv2.INTER_CUBIC),0,1)

#### Images are shown here side-by-side to compare
sub3(p1,p2,p3)
sub3(nod(lx,10),nod(lx,30),nod(lx,50))

print('psnrv',tf.image.psnr(tim,p1,max_val=1).numpy())
print('psnrv',tf.image.psnr(tim,p2,max_val=1).numpy())
print('psnrv',tf.image.psnr(tim,p3,max_val=1).numpy())

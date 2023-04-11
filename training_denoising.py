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



#### Loss Functions

def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def PLoss(y_true, y_pred):
    return (100 - tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0)))/100
  
    
def hu(x,fa):
    return tf.image.adjust_hue(x,fa )

def con(x,fa):
    return tf.image.adjust_contrast(x,fa )


def gm(x,fa):
    return tf.image.adjust_gamma(x,fa )

def brt(x,fa):
    return tf.image.adjust_brightness(x,fa )

def aug(x):
    a1=tf.image.flip_left_right(x )
    a2=tf.image.flip_up_down(x )
    a3=tf.image.transpose(x )
    return a1,a2,a3


def mce(y_true, y_pred):
            
    evas = K.abs(y_pred - y_true)
    evas = K.mean(evas, axis=-1)
        
    return evas

### Final Loss
def ca1(y_true, y_pred):
    
    y_pred=y_pred
            
    er = .25*mce( y_true,y_pred)+ .85*SSIMLoss(y_true, y_pred )+ 0.3*PLoss(y_true, y_pred)  
    return  er


### Data loading

x1 = np.load('./noisy.npy')
y1 = np.load('./target.npy')

X_train, X_val, y_train, y_val = train_test_split(np.float32(x1), np.float32(y1),  test_size=0.1, random_state=42)

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

train_dataset = DataGenerator(X_train, y_train, 8) ### Here datagenerator used to load 8 images at a time for training
val_dataset = DataGenerator(X_val, y_val, 8) ### Here datagenerator used to load 8 images at a time for validation


#### Creating the model for training

### Modules for training
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

#### Building custom optimizer for training

def gc(gradis):
    cgrads = []
    for grad in gradis:
        rank = len(grad.shape)
        if rank > 1:
            grad -= tf.reduce_mean(grad, axis=list(range(rank-1)))
        cgrads.append(grad)
    return cgrads

from tensorflow.keras.optimizers import Adam
class GCRMSprop(Adam):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads


optim = GCRMSprop(learning_rate=1e-4)

#### Compiling the model 

model.compile(loss=ca1, optimizer=optim, metrics=["accuracy"])

#### Starting training using gradient tape
import time

epochs = 1000
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch+1,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x_batch_train = np.asarray(tf.image.resize(x_batch_train,[128, 128]))
            x_batch_train = x_batch_train/255
            y_batch_train = np.asarray(tf.image.resize(y_batch_train,[128, 128]))
            y_batch_train = y_batch_train/255
            
            logits = model(x_batch_train, training=True)
            xi=x_batch_train
            yi=y_batch_train
 
            loss_value =   (ca1(yi, logits)) 
    
        grads =   (tape.gradient(loss_value, model.trainable_weights))
        optim.apply_gradients(zip(grads, model.trainable_weights))


    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        x_batch_val = x_batch_val/255
        y_batch_val = y_batch_val/255
        val_logits = model(x_batch_val, training=False)
        loss_val =  (ca1(  y_batch_val  , val_logits) ) 
    
    print('val_loss: ', tf.reduce_mean(loss_val).numpy())
    current = tf.reduce_mean(loss_val).numpy()
   
    if epoch == 0:
        lowest = tf.reduce_mean(loss_val).numpy()
       
    if current <= lowest:
        model.save_weights('./weights/noise_weight.h5') ### Check with the lowest validation loss and update the weight file
        lowest = current
    
    print("Time taken: %.2fs" % (time.time() - start_time))



 
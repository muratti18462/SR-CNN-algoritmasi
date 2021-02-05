#srcnn murat ilk kısım optimizasyon yok
def save_images(data_path,save_path,size1,size2):
  data_path = glob.glob(data_path)

  for i in range(len(data_path)):
    fn = data_path[i].split("\\")[-1]
    img = cv2.imread(data_path[i],cv2.IMREAD_UNCHANGED)
    n_img = cv2.resize(img,(size1,size2),cv2.INTER_AREA)
    cv2.imwrrite(save+"\{}.png".format(str(fn)),n_img)
    print("{}.dosya kaydediliyor".format(str(i)))
  print("işlem bitti.")

def savingLR(data_path,save_path,factor):
   data_path = glob.glob(data_path)
   for i in range(len(data_path)):
    fn = data_path[i].split("\\")[-1]
    img = cv2.imread(data_path[i],cv2.IMREAD_UNCHANGED)
    width = img.shape[1]
    height = img.shape[0]
    newWidth = int(width/factor)
    newHeight = int(height/factor)
    img = cv2.resize(img,(newWidth,newheight),cv2.INTER_CUBIC)
    LRImg = cv2.resize(img,(width,height),cv2.INTER_CUBIC)
    print("{}.dosya kaydediliyor".format(str(i)))
  print("işlem bitti.")

def crop_saving(data_path,save_path):
  data_path = glob.glob(data_path)
  print(data_path)

  for i in range(len(data_path)):
    fn = data_path[i].split("\\")[-1].split('.')[0]
    img = cv2.imread(data_path[i],cv2.IMREAD_UNCHANGED)
    print("{}.resim kırpılıyor.".format(str(i)))
    counter = 1
    for j in range(8):
      for k in range(8):
        crop_img = img[j*64:(j+1)*64,k*64:(k+1)*64].copy()
        cv2.imwrite(save_path+"\{}_{}.png".format(str(fn),str(counter)),crop_img)
        counter=counter+1

import pandas as pd
import numpy as np
import cv2
import glob
import os

data_path = r"C:\Users\Casper\Desktop\sss\train_data\*.png"
label_path = r"C:\Users\Casper\Desktop\sss\label_data\*.png"

data_path = glob.glob(data_path)
label_path = glob.glob(label_path)
data_arr = []
label_arr = []

for i in range(len(data_path)):
  img = cv2.imread(data_path[i],cv2.IMREAD_UNCHANGED)
  img = cv2.resize(img,(64,64),cv2.INTER_CUBİC)
  img_YCrCb = cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
  img_YCrCb = img_YCrCb/255
  img_y = img_YCrCb[:,:,0]
  data_arr.append(img_y)

a= np.array([np.array(x) for x in data_arr])

np.save("x.npy",a)

for i in range(len(label_path)):
  img = cv2.imread(data_path[i],cv2.IMREAD_UNCHANGED)
  img = cv2.resize(img,(64,64),cv2.INTER_CUBİC)
  img_YCrCb = cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
  img_YCrCb = img_YCrCb/255
  img_y = img_YCrCb[:,:,0]
  data_arr.append(img_y)

b= np.array([np.array(x)for x in label_arr])

np.save("y.pny",b)
x = np.load("x.pny")
y = np.load("y.npy")
y = x.reshape(-1,64,64,1)
x = y.reshape(-1,64,64,1)

import math
from sklear.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam



def model():

  SRCNN=Sequential()

  SRCNN.add(Conv2D(filters=128,kernel_size=(9,9),kernel_initializer ='random_uniform',
                   activation='relu',padding = 'same',use_bias=True,input_shape=(64,64,1)))
  SRCNN.add(Conv2D(filters=64,kernel_size=(3,3),kernel_initializer ='random_uniform',
                   activation='relu',padding = 'same',use_bias=True))
  SRCNN.add(Conv2D(filters=1,kernel_size=(5,5),kernel_initializer ='random_uniform',
                   activation='relu',padding = 'same',use_bias=True))
   
  optimizer = Adam(lr=0.0001)

  SRCNN.compile(optimizer = optimizer,loss = 'mean_squared_error',metrics=['accuracy'])

  return SRCNN

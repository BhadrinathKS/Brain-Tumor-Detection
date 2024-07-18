# Brain Tumor Detection
# Bhadrinath Kolluru
# July 18th, 2024

from google.colab import drive
drive.mount('/content/drive')

# this creates a symbolic link so that now the path /content/gdrive/My\ Drive/ is equal to /mydrive
!ln -s /content/drive/My\ Drive/ /Gdrive
!ls /Gdrive

# Import Libraries
import os, shutil
import cv2
import glob
import imutils
import xml.etree.ElementTree as ET
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

# Import Keras (VGG-19 Machine Learning Model)
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg19 import VGG19
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Renaming all Files

# Yes Directory
tumerous = '/Gdrive/MachineLearningProject/images/datasets/yes/'
countY = 1

for filename in os.listdir(tumerous):
    source = tumerous + filename
    destination = tumerous + "Y_" +str(countY)+".jpg"
    os.rename(source, destination)
    countY+=1
print("All files are renamed in the Yes Directory!\n")

print(". . .\n")

# No Directory 
non_tumerous = '/Gdrive/MachineLearningProject/images/datasets/no/'
countN = 1

for filename in os.listdir(non_tumerous):
    source = non_tumerous + filename
    destination = non_tumerous +"N_" +str(countN)+".jpg"
    os.rename(source, destination)
    countN+=1
print("All files are renamed in the No Directory")

# EDA (Exploratory Data Analysis)

Ylist = os.listdir("/Gdrive/MachineLearningProject/images/datasets/yes/")
number_files_yes = len(Ylist)
print("Yes Files: "+str(number_files_yes)) 


Nlist = os.listdir("/Gdrive/MachineLearningProject/images/datasets/no/")
number_files_no = len(Nlist)
print("No Files: "+str(number_files_no))

# Plot Of Original Sample

data = {'tumerous': number_files_yes, 'non-tumerous': number_files_no}

typex = data.keys()
values = data.values()

fig = plt.figure(figsize=(5,7))
plt.bar(typex, values, color="red")
plt.xlabel("Data")
plt.ylabel("# of Tumerous Images")
plt.title("Count of Tumerous Brain Images")

# Create Timer 
def timing(sec_elapsed): 
    h = int(sec_elapsed / (60*60))
    m = int(sec_elapsed % (60*60) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{s}"  

# 155(61%), 98(39%)
# There is an imbalance in the sample so it is important to create more images to increase accuracy of the model

def augmented_data(file_dir, n_gererated_samples, save_to_dir):
    data_gen = ImageDataGenerator(rotation_range=10,
                       width_shift_range=0.1,
                       height_shift_range=0.1,
                       shear_range=0.1,
                       brightness_range=(0.3, 1.0),
                       horizontal_flip=True,
                       vertical_flip=True,
                       fill_mode='nearest')
    for filename in os.listdir(file_dir):
      image = cv2.imread(file_dir + '/' + filename)
      image = image.reshape((1,) + image.shape)
      save_prefix = 'aug_' + filename[:-4]
      i = 0
      for batch in data_gen.flow(x = image, batch_size = 1, save_to_dir = save_to_dir, save_prefix = save_prefix, save_format = "jpg"):
        i+=1
      if i>n_gererated_samples:
          break

      # Augmentation Start
import time
start_time = time.time()

yes_path = '/Gdrive/MachineLearningProject/images/datasets/yes/'
no_path = '/Gdrive/MachineLearningProject/images/datasets/no/'
augmented_data_path = '/Gdrive/MachineLearningProject/data/augmented_data/'

augmented_data(file_dir = yes_path, n_gererated_samples=6, save_to_dir=augmented_data_path+'yes')
augmented_data(file_dir = no_path, n_gererated_samples=9, save_to_dir=augmented_data_path+'no')

end_time = time.time()
execution_time = end_time - start_time
print(timing(execution_time))

# Post-Augmentation Results

def data_summary(main_path)
  yes_path = "/Gdrive/MachineLearningProject/images/datasets/yes/"
  no_path = "/Gdrive/MachineLearningProject/images/datasets/no/"

  n_pos = len(os.listdir(yes_path))
  n_neg = len(os.listdir(no_path))
  n = (n_pos + n_neg)
  pos_per = (n_pos*100/n)
  neg_per = (n_neg*100/n)

  print(f"Number of samples: {n}")
  print("Number of positive samples: "+f"{n_pos} "+"("+{pos_per}%+")")
  print("Number of negative samples: "+f"{n_neg} "+"("+{neg_per}%+")")

# EDA (Exploratory Data Analysis) - Post Augmentation

Ylist = os.listdir("/Gdrive/MachineLearningProject/images/datasets/yes/")
number_files_yes = len(Ylist)
print("Yes Files: "+str(number_files_yes)) 


Nlist = os.listdir("/Gdrive/MachineLearningProject/images/datasets/no/")
number_files_no = len(Nlist)
print("No Files: "+str(number_files_no))

# New Plot 

data = {'tumerous': number_files_yes, 'non-tumerous': number_files_no}

typex = data.keys()
values = data.values()

fig = plt.figure(figsize=(5,7))
plt.bar(typex, values, color="red")
plt.xlabel("Data")
plt.ylabel("# of Tumerous Images")
plt.title("Count of Tumerous Brain Images")

# Convert BGR to GRAY
# GaussianBlur
# Threshold
# Erode
# Dilate
# Find Contours

def crop_brain_tumor(image, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    thres = cv2.threshhold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thres = cv2.erode(thres, None, iterations = 2)
    thres = cv2.dilate(thres, None, iterations = 2)

    cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts =  imutils.grab_contours(cnts)
    c = max(cnts, key = cv2.contourArea)
    

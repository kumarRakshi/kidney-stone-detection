import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
import random
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2 
from skimage.feature import hog 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing import image
def custom_Image_preprocessing(image_data, target_size=(150, 150)):
    img = image.array_to_img(image_data, data_format='channels_last')
    img = img.resize(target_size)  # Resize the image if needed
    img_arr = image.img_to_array(img)
    img_arr = img_arr * 1./255
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr
def predict(imgpath):
    img = image.load_img(imgpath,target_size=(150, 150))
    image_preprocess = custom_Image_preprocessing(img)
    model = load_model('kidney_stone_detection_model.h5',compile = False)
    result = model.predict(image_preprocess)
    print("result==",result)
    print(result[0][0])
    if result[0][0] > 0.5:
        return 'Kidney Stone Detected (Positive)',round(result[0][0]*100,2),'%'
    else:
        return 'No Kidney Stone Detected  (Negative)',round(result[0][0]*100,2),'%'
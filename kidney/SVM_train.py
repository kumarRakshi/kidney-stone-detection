import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.utils import load_img
import random
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2 
from skimage.feature import hog 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
import sklearn.externals
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing import image
import joblib
#Defining a function to read images from the train and test folders
def process():
    
    def read_images(path):
        images_list = []
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path,filename))
            if img is not None:
                images_list.append(img)
        return images_list
    #Reading train images from the normal and stone folders
    train_normal = read_images('CT_images/Train/Normal')
    train_stone = read_images('CT_images/Train/Stone')
    #Creating a list of labels for training 
    labels = ['Normal' for item in train_normal] + ['Stone' for item in train_stone]
    #Defining a function for HOG feature extraction
    def extract_features(images):
        feature_list = []
        for img in images:
            fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), 
                                cells_per_block=(1, 1), visualize=True, channel_axis=2)
            # Resize the HOG features to a fixed size
            fd = np.resize(fd, (2400, 1))
            # Flatten the array to 2 dimensions
            fd = fd.flatten()
            feature_list.append(fd)
        return feature_list
    #Extracting the HOG features from both normal and stone images
    feature_list_normal = extract_features(train_normal)
    feature_list_stone = extract_features(train_stone)
    print(len(feature_list_normal))
    print(len(feature_list_stone))
    #Combining the features for both classes
    features = feature_list_normal + feature_list_stone
    #Reading test images from the normal and stone folders
    test_normal = read_images('CT_images/Test/Normal')
    test_stone = read_images('CT_images/Test/Stone')
    #Creating a list of labels for testing 
    test_labels = ['Normal' for item in test_normal] + ['Stone' for item in test_stone]
    #Creating a Feature Vector for Test Set
    test_feature_list_normal = extract_features(test_normal)
    test_feature_list_stone = extract_features(test_stone)
    print(len(test_feature_list_normal))
    print(len(test_feature_list_stone))
    #Combining the features for both classes
    test_features = test_feature_list_normal + test_feature_list_stone
    #Splitting the data into train and valid sets
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.2, random_state=0)
    # Print the shape of the first element in the X_train array
    print(X_train[0].shape)

    # Print the shape of the second element in the X_train array
    print(X_train[1].shape)

    # Print the shape of the last element in the X_train array
    print(X_train[-1].shape)
    # Training a SVM Model
    svc = SVC(kernel='rbf', C=1, gamma='auto')
    svc.fit(X_train, y_train)
    # Predicting the Test Set
    y_pred = svc.predict(X_valid)
    #Calculating the accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    print("Accuracy : ", accuracy)
    svm_cm = confusion_matrix(y_valid, y_pred)

    sns.heatmap(svm_cm, annot=True, fmt='d', cmap='BuPu')

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    plt.show()


    # Save the model to a file
    joblib.dump(svc, 'svc.pkl')

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
from sklearn.metrics import confusion_matrix
def process(path):
    folder_path = str(path)+"/Train/"
    filenames = []
    categories = []
    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                filenames.append(os.path.join(category, filename))
                categories.append(category)
    df = pd.DataFrame({'filename': filenames,'category': categories})
    model = Sequential()
    #Adding convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Adding a second convolutional layer
    model.add(Conv2D(64, (3, 3) , activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # this converts our feature maps to 1D feature vectors
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2)) # as we have binary class i.e stone and normal so value is 2
    model.add(Activation('sigmoid')) #sigmoid for binary class classification
    model.summary()
    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
    callbacks = [earlystop, learning_rate_reduction]
    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    train_datagen = ImageDataGenerator(rotation_range=15,rescale=1./255,shear_range=0.1,zoom_range=0.2,horizontal_flip=True,width_shift_range=0.1,height_shift_range=0.1)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_dataframe(train_df, "./CT_Images/Train/", x_col='filename',y_col='category',target_size=(150,150),class_mode='categorical',batch_size=15)
    validation_generator = validation_datagen.flow_from_dataframe(validate_df, "./CT_Images/Train/", x_col='filename',y_col='category',target_size=(150,150),class_mode='categorical',batch_size=15)
    # Compiling the model
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    history = model.fit( train_generator, epochs=50,validation_data=validation_generator,validation_steps=validate_df.shape[0]//15,steps_per_epoch=train_df.shape[0]//15,callbacks=callbacks)
    # Plotting the training history
    plt.figure(figsize=(12, 8)) 
    # Plot training & validation accuracy values
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    # Plot training & validation loss values
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.show()
    # Saving the Model
    model.save('kidney_stone_detection_model.h5')
    test_folder_path = str(path)+"/Test"
    test_filenames = []
    test_categories = []
    for category in os.listdir(test_folder_path):
        test_category_path = os.path.join(test_folder_path, category)
    if os.path.isdir(test_category_path):
        for filename in os.listdir(test_category_path):
            test_filenames.append(os.path.join(category, filename))
            test_categories.append(category)
    test_df = pd.DataFrame({'filename': test_filenames, 'category': test_categories})
    test_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_gen.flow_from_dataframe(test_df, str(path)+"/Test/", x_col='filename',y_col='category',target_size=(150,150),class_mode='categorical',batch_size=15,shuffle=False)
    steps = np.ceil(test_df.shape[0] / 15)
    predict = model.predict(test_generator, steps=steps)
    test_df['predicted category'] = np.argmax(predict, axis=-1)
    test_df['predicted category'] = test_df['predicted category'].replace({ 1: 'Stone', 0: 'Normal' })
    plt.figure(figsize=(12, 4))
    # Plot the first bar plot (predicted category)
    plt.subplot(1, 2, 1)
    test_df['predicted category'].value_counts().plot.bar()
    plt.title('Predicted Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    # Plot the second bar plot (actual category)
    plt.subplot(1, 2, 2)
    test_df['category'].value_counts().plot.bar()
    plt.title('Actual Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plots
    plt.show()
    cm = confusion_matrix(test_df['category'], test_df['predicted category'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    model.evaluate(test_generator)
#process("./CT_images")
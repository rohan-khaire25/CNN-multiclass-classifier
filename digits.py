# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:42:01 2020

@author: Rohan khaire
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv('C:/digits/train.csv')
test = pd.read_csv('C:/digits/test.csv')

x_train = train.drop(labels=['label'], axis = 1)
y_train = train['label']

x_train = x_train/255
test = test/255

x_train = x_train.values.reshape(-1, 28, 28, 1)
xtest = test.values.reshape(-1, 28, 28, 1)

from keras.utils.np_utils import to_categorical 
y_train = to_categorical(y_train, num_classes = 10)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state = 2)

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (5,5), activation='relu', input_shape = (28, 28, 1)),
                                   tf.keras.layers.Conv2D(32, (5,5), activation='relu'),
                                   tf.keras.layers.Dropout(0.3),
                                   tf.keras.layers.MaxPooling2D(2,2),                    
                                   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                   tf.keras.layers.MaxPooling2D(2,2),
                                   tf.keras.layers.Dropout(0.3),
                                   tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(512, activation='relu'),
                                   tf.keras.layers.Dropout(0.3),
                                   tf.keras.layers.Dense(10, activation='softmax')])

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer = RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics = ['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zoom_range=0.2,
                             rotation_range=40,
                             width_shift_range = 0.1,
                             height_shift_range = 0.1)

datagen.fit(x_train)
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.callbacks import ReduceLROnPlateau
#lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.01, epsilon=0.0001, patience=3, verbose=1)

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 86),
                              epochs = 30,
                              steps_per_epoch = 300,
                              validation_data = (x_val, y_val),
                              validation_steps = 50,
                              verbose = 1,
                              callbacks = [lr_reduce])


pred = model.predict(xtest)
y_classes = pred.argmax(axis=-1)
res = pd.DataFrame()
res['ImageId'] = list(range(1,28001))
res['Label'] = y_classes
res.to_csv("output.csv", index = False)


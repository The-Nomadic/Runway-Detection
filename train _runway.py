import keras, os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from tensorflow.python.util import deprecation
from tensorflow.keras import backend
from tensorflow.keras.applications.resnet50 import preprocess_input
# deprecation._PRINT_DEPRECATION_WARNINGS = False

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from tensorflow.keras.models import load_model
print("import done")


def vgg_16(weights = 'imagenet', include_top = True):

    model = Sequential()
    model.add(Conv2D(input_shape = (24,24,1), filters = 32, kernel_size = (3,3), padding = "same", activation = "relu"))

    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = "same", activation = "relu"))

    model.add(MaxPool2D(pool_size = (2,2), strides = (2,2),padding='same'))

    model.add(Flatten())
   
    model.add(Dense(units = 128, activation = "relu"))
    
    model.add(Dense(units = 1, activation = "sigmoid"))

    # model.summary()
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model
  
print(vgg_16().summary())


vgg_16_model = vgg_16()

trdata = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

traindata = trdata.flow_from_directory(directory="dataset/train",target_size=(24,24),
                                            color_mode='grayscale',
                                            batch_size = 32,
                                            class_mode = 'binary')
tsdata = ImageDataGenerator(rescale = 1./255)
testdata = tsdata.flow_from_directory(directory="dataset/val", target_size=(24,24),
                                            color_mode='grayscale',
                                            batch_size = 32,
                                            class_mode = 'binary')


vgg_16_model = vgg_16()

checkpoint = tf.keras.callbacks.ModelCheckpoint("runway_weights.hdf5",
                                                monitor="val_loss",
                                                verbose = 1,
                                                save_best_only = False,
                                                save_weights_only = False,
                                                mode= "auto",
                                                save_freq= "epoch",
                                                options=None)

early = EarlyStopping(monitor='val_accuracy',
                      min_delta=0,
                      patience=40,
                      verbose=1,
                      mode='auto')

vgg_16_model.fit(traindata,
                          steps_per_epoch = 16,
                          epochs = 15,
                          validation_data = testdata,
                          validation_steps = 20,
                          callbacks = [checkpoint,early])

vgg_16_model.save("runway_weights.hdf5")




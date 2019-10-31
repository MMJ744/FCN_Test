import cv2
import PIL
import os
import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense


def create_model():
    # Channels first tells the pooling layer to use the (Height, Width, Depth) format instead of the (Depth, Height, Width)
    data_format="channels_first"

    # A sequential model is a basic model structure where the layers are stacked layer by layer.
    # Another option with keras is a functional model, layers can be connected to literally any other layer within the model.
    model = Sequential()
    # A convolutional layer slides a filter over the image which is fed to the activation layer so the model can learn
    # features and activate when they see one of these visual features. Only activated features are carried over to the
    # next layer.
    model.add(Convolution2D(32, (3, 3), input_shape=(255, 255, 3)))
    # Relu maps all negative values to 0 and keeps all positive values.
    model.add(Activation('relu'))
    # A pooling layer reduces the dimensions of the image but not the depth. The computation is faster and less image data
    # means less chance of over fitting.
    model.add(MaxPooling2D(data_format=data_format, pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(data_format=data_format, pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(data_format=data_format, pool_size=(2, 2)))

    # Squashes the output of the previous layer to an array with 1 dimension
    model.add(Flatten())
    # A dense layer's takes n num of inputs and is connected to every output by the weights it produces.
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Fully connected layer or Dense layer in keras, with the number of classes that the network will be able to predict.
    model.add(Dense(1))
    # Sigmoid is used because it exists between 0 and 1 and we want to give a prediction between 0 and 1.
    model.add(Activation('sigmoid'))

    model.compile(optimizer='rmsprop',              # Performs gradient descent, finding the lowest error value
                  loss='binary_crossentropy',       # Cross entropy measures the performance of each prediction made by the network
                  metrics=['accuracy'])

    return model

batch_size = 16
train_size = 2000
test_size = 500
train = []
truth = []
path = 'data/train/'
for file in os.listdir(path):
    img = image.imread(path+file)
    train.append(img)
print(str(len(train))+ " images loaded for training")
path = 'data/train_truth/'
for file in os.listdir(path):
    img = image.imread(path+file)
    truth.append(img)
print(str(len(truth))+ " Truth images loaded for training")
model = create_model()

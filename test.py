import cv2
import PIL
import random
import os
import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import EarlyStopping
from matplotlib import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Lambda
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Deconvolution2D, Conv2DTranspose, Conv2D
from keras.layers.merge import concatenate
from keras import backend

def cnn_model():
    # Channels first tells the pooling layer to use the (Height, Width, Depth) format instead of the (Depth, Height, Width)
    data_format="channels_first"
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(data_format=data_format, pool_size=(2, 2)))
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(data_format=data_format, pool_size=(2, 2)))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(data_format=data_format, pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='rmsprop',              # Performs gradient descent, finding the lowest error value
                  loss='binary_crossentropy',       # Cross entropy measures the performance of each prediction made by the network
                  metrics=['accuracy'])

    return model

def create_model():
    # Channels first tells the pooling layer to use the (Height, Width, Depth) format instead of the (Depth, Height, Width)
    data_format="channels_first"
    # A sequential model is a basic model structure where the layers are stacked layer by layer.
    # Another option with keras is a functional model, layers can be connected to literally any other layer within the model.
    model = Sequential()
    # A convolutional layer slides a filter over the image which is fed to the activation layer so the model can learn
    # features and activate when they see one of these visual features. Only activated features are carried over to the
    # next layer.
    model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3),))
    # Relu maps all negative values to 0 and keeps all positive values.
    model.add(Activation('relu'))
    # A pooling layer reduces the dimensions of the image but not the depth. The computation is faster and less image data
    # means less chance of over fitting.
    model.add(MaxPooling2D(data_format=data_format, pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(data_format=data_format, pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(data_format=data_format, pool_size=(2, 2)))
    model.add(Conv2DTranspose(1, (3, 3), output_shape=(256,256)))
    model.compile(optimizer='rmsprop',              # Performs gradient descent, finding the lowest error value
                  loss='binary_crossentropy',       # Cross entropy measures the performance of each prediction made by the network
                  metrics=['accuracy'])
    print(model.summary())
    return model

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        backend.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return backend.mean(backend.stack(prec), axis=0)


def unet_model():
    inputs = Input((256,256,3))
    s = Lambda(lambda x: x / 255)(inputs)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()
    return model

model = unet_model()
batch_size = 16
train_size = 2000
test_size = 500
train = []
truth = []
path = 'data/train/'
for file in os.listdir(path):
    img = image.imread(path+file)
    train.append(img/255)
print(str(len(train))+ " images loaded for training")
print(np.asarray(train).shape)
path = 'data/train_truth/'
for file in os.listdir(path):
    img = image.imread(path+file)
    truth.append(img/255)
print(str(len(truth))+ " Truth images loaded for training")
print(np.asarray(truth).shape)
x, y = zip(*random.sample(list(zip(train, truth)), train_size+test_size))
x_train = np.asarray(x[:train_size])
#x_train = x_train.reshape((1,)+ x_train.shape)
x_test = np.asarray(x[train_size:])
y_train = np.asarray(y[:train_size])
y_train = y_train.reshape(y_train.shape + (1,))
y_test = np.asarray(y[train_size:])
y_test = y_test.reshape(y_test.shape + (1,))
print(y_train.shape)
print(y_test.shape)
callbacks = [EarlyStopping(patience=10, verbose=1)]
#model = create_model()
model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=5,validation_data=(x_test,y_test),callbacks = callbacks)

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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
from tensorflow.python.client import device_lib

print("-------------")

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

def dice_coef(y_true, y_pred, smooth=1):
  intersection = backend.sum(y_true * y_pred, axis=[1,2,3])
  union = backend.sum(y_true, axis=[1,2,3]) + backend.sum(y_pred, axis=[1,2,3])
  dice = backend.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

def unet_model():
    inputs = Input((256, 256, 3))
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
dir = sorted(os.listdir(path))
for file in dir:
    img = image.imread(path+file)
    train.append(img/255)
print(str(len(train))+ " images loaded for training")
print(np.asarray(train).shape)
path = 'data/train_truth/'
dir = sorted(os.listdir(path))
for file in dir:
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
callbacks = [EarlyStopping(patience=10, verbose=1,monitor='val_loss'),ReduceLROnPlateau(patience=5, verbose=1,monitor='val_loss')]
WEIGHTS_FILE = 'test.h5'
if(False):
    model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=1,validation_data=(x_test,y_test),callbacks = callbacks)
    model.save_weights(WEIGHTS_FILE)
else:
    model.load_weights(WEIGHTS_FILE)
#print(model.evaluate(x_test,y_test))
print("--------------")
img = np.asarray(train[120])
#plt.imshow(img)
#plt.show()
plt.imshow(truth[120])
plt.show()
img = img.reshape((1,) + img.shape)
print('**'+str(img.shape))
output = model.predict(img)[0]
print(output.min())
print(output.max())
outputround = output.copy()
cutoff = 0.6
outputround[outputround > cutoff] = 1
outputround[outputround <= cutoff] = 0
outputround = cv2.merge((outputround,outputround,outputround))
print(outputround.min())
print(outputround.max())
print(outputround.shape)

output = cv2.merge((output,output,output))
plt.imshow(outputround, interpolation='nearest')
plt.show()
plt.imshow(output, interpolation='nearest')
plt.show()

import cv2
import PIL
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import image
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Lambda
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Deconvolution2D, Conv2DTranspose, Conv2D
from keras.layers.merge import concatenate
from keras import backend as K
from tensorflow.python.client import device_lib
from sklearn.utils import class_weight


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def iou(true, pred):
    #cutoff = 0.5
    #true = true > cutoff
    #pred = pred > cutoff
    #intersection = true & pred
    #union = true | pred
    #iou_score = backend.sum(intersection) / backend.sum(union)
    score = tf.compat.v1.metrics.mean_iou(true, pred, 2)
    print(score)
    print(type(score))
    return score

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
  union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def dice_coef(y_true, y_pred, smooth=1):
  print(y_true.shape)
  print(y_pred.shape)
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
  union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
  dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[iou_coef,f, precision,recall])
    #model.summary()
    return model


def showImg(id):
    img = np.asarray(train[id])
    plt.imshow(truth[id])
    plt.show()
    img = img.reshape((1,) + img.shape)
    output = model.predict(img)[0]
    cutoff = 0.5
    output[output > cutoff] = 1
    output[output <= cutoff] = 0
    output = cv2.merge((output, output, output))
    plt.imshow(output, interpolation='nearest')
    plt.show()

def findWeights(ydata):
    weights = class_weight.compute_class_weight('balanced',np.unique(ydata),ydata.flat)

def load_data():
    train = []
    truth = []
    path = 'data/train/'
    dir = sorted(os.listdir(path))
    for file in dir:
        img = image.imread(path + file)
        train.append(img / 255)
    print(str(len(train)) + " images loaded for training")
    path = 'data/train_truth/'
    dir = sorted(os.listdir(path))
    for file in dir:
        img = image.imread(path + file)
        truth.append(img / 255)
    print(str(len(truth)) + " Truth images loaded for training")
    return train, truth


if __name__ == "__main__":
    train_size = 2000
    test_size = 500
    model = unet_model()
    train, truth = load_data()
    x, y = zip(*random.sample(list(zip(train, truth)), train_size + test_size))
    x_train = np.asarray(x[:train_size])
    # x_train = x_train.reshape((1,)+ x_train.shape)
    x_test = np.asarray(x[train_size:])
    y_train = np.asarray(y[:train_size])
    y_train = y_train.reshape(y_train.shape + (1,))
    y_test = np.asarray(y[train_size:])
    y_test = y_test.reshape(y_test.shape + (1,))
    weight = {0: 1.0, 1: 10.0}
    callbacks = [EarlyStopping(patience=10, verbose=1, monitor='val_loss'),
                 ReduceLROnPlateau(patience=5, verbose=1, monitor='val_loss')]
    WEIGHTS_FILE = 'class_weights.h5'
    if (True):
        model.fit(x=x_train, y=y_train, batch_size=32, epochs=200, validation_data=(x_test, y_test),
                  callbacks=callbacks)
        model.save_weights(WEIGHTS_FILE)
    else:
        model.load_weights(WEIGHTS_FILE)
    x = x_train[50]
    x = x.reshape((1,) + x.shape)
    out = model.predict(x)[0]
    print(model.evaluate(x_test, y_test))  # (loss,accuracy)
    print("--------------")
    showImg(2121)
    showImg(200)
    showImg(100)
    showImg(25)
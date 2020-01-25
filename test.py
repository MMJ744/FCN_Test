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
import keras.preprocessing as preprocessing
from tensorflow.python.client import device_lib
from sklearn.utils import class_weight
import  models

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


def showImg(id):
    img = np.asarray(test[id])
    real = test_truth[id]
    real = cv2.merge((real, real, real))
    plt.imshow(real)
    plt.show()
    img = img.reshape((1,) + img.shape)
    output = model.predict(img)[0]
    cutoff = 0.5
    output[output > cutoff] = 1
    output[output <= cutoff] = 0
    output = cv2.merge((output, output, output))
    plt.imshow(output, interpolation='nearest')
    plt.show()


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
    train_size = 2250
    test_size = 750
    model = models.relu()
    WEIGHTS_FILE = 'weights/relu-4-0.h5'
    train, truth = load_data()
    x, y = train, truth
    x_train = np.asarray(x[:train_size])
    # x_train = x_train.reshape((1,)+ x_train.shape)
    x_test = np.asarray(x[train_size:])
    y_train = np.asarray(y[:train_size])
    y_train = y_train.reshape(y_train.shape + (1,))
    y_test = np.asarray(y[train_size:])
    y_test = y_test.reshape(y_test.shape + (1,))
    weight = {0: 1.0, 1: 10.0}
    callbacks = [EarlyStopping(patience=15, verbose=1, monitor='val_loss'),
                 ReduceLROnPlateau(patience=5, verbose=1, monitor='val_loss')]
    if (True):
        model.fit(x=x_train, y=y_train, batch_size=32, epochs=150, validation_data=(x_test, y_test),callbacks=callbacks )#
        model.save_weights(WEIGHTS_FILE)
    else:
        model.load_weights(WEIGHTS_FILE)
    test_truth = []
    path = 'data/test_truth/'
    dir = sorted(os.listdir(path))
    for file in dir:
        img = image.imread(path + file)
        test_truth.append(img / 255)
    path = 'data/test/'
    dir = sorted(os.listdir(path))
    test = []
    for file in dir:
        img = image.imread(path + file)
        test.append(img / 255)
    test = np.asarray(test)
    test_truth = np.asarray(test_truth)
    test_truth = test_truth.reshape(test_truth.shape + (1,))
    print(model.evaluate(test,test_truth))
    showImg(140)
    showImg(30)
    showImg(260)
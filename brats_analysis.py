import cv2
import os
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import image
import models
import matplotlib.pyplot as plt
import nibabel


def load_data():
    train = []
    truth = []
    path = 'data/LGG/'
    dir = sorted(os.listdir(path))
    seg = 0
    modalitys = []
    for folder in dir:
        fdir = sorted(os.listdir(path + folder))
        modalitys = []
        for file in fdir:
            img = nibabel.load(path + folder + '/' + file)
            data = img.get_fdata()
            data = np.swapaxes(data, 1, 2)
            data = np.swapaxes(data, 0, 1)
            data = np.float32(data)
            if 'seg' in file:
                seg = data
            else:
                modalitys.append(data)
        for modal in modalitys:
            for x, y in zip(modal, seg):
                if np.max(x) > 0:
                    train.append(x)
                    truth.append(y)
    return np.asarray(train), np.asarray(truth)


def cropping_tests():
    x = []
    y = []
    path = 'data/brats_test/'
    dir = sorted(os.listdir(path))
    for file in dir:
        img = nibabel.load(path + file)
        data = img.get_fdata()
        print("MAX ***** " + str(np.max(data)))
        print(img.header)
        if 'seg' in file:
            y.append(data)
        else:
            x.append(data / 255)
    test = x[0]
    print(test.shape)
    new = np.zeros((240, 240))
    noneZero = False
    c = 0
    c1 = 0
    for a in test:
        c2 = 0
        for b in a:
            new[c1][c2] = np.max(b)
            c2 += 1
        c1 += 1

    print(new.shape)
    for i in range(240):
        if np.max(new[i]) != 0:  # 239-i
            print(i)
            break
    for i in range(240):
        if np.max(new[239 - i]) != 0:  # 239-i
            print(i)
            break
    new = np.swapaxes(new, 0, 1)
    for i in range(240):
        if np.max(new[i]) != 0:  # 239-i
            print(i)
            break
    for i in range(240):
        if np.max(new[239 - i]) != 0:
            print(i)
            break
    new = np.swapaxes(new, 0, 1)
    cropped = new[48:192, 40:221]
    plt.imshow(new)
    plt.show()
    plt.imshow(cropped)
    plt.show()
    print("------")
    for i in range(cropped.shape[0]):
        if np.max(cropped[i]) != 0:  # 239-i
            print(i)
            break
    for i in range(cropped.shape[0]):
        if np.max(cropped[cropped.shape[0] - 1 - i]) != 0:  # 239-i
            print(i)
            break
    cropped = np.swapaxes(cropped, 0, 1)
    for i in range(cropped.shape[0]):
        if np.max(cropped[i]) != 0:  # 239-i
            print(i)
            break
    for i in range(cropped.shape[0]):
        if np.max(cropped[cropped.shape[0] - 1 - i]) != 0:
            print(i)
            break
    print(cropped.shape)
    print(np.max(cropped))


model = models.brats()
print(model.summary())
scans, segment = load_data()
print(scans.shape)
print(segment.shape)
plt.imshow(scans[3000])
plt.show()
plt.imshow(segment[3000])
plt.show()
#print(np.max(scans))
#print(np.average(scans))
#print(np.min(scans))
#print(np.max(segment))
#print(np.average(segment))
#print(np.min(segment))
#print(len(scans))
#print(len(segment))
scans = scans.reshape(scans.shape + (1,))
segment = segment.reshape(segment.shape + (1,))
train_size = int(len(scans) * 0.8)
x_train = scans[:train_size]
x_test = scans[train_size:]
y_train = segment[:train_size]
y_test = segment[train_size:]
callbacks = [EarlyStopping(patience=15, verbose=1, monitor='val_loss'),
             ReduceLROnPlateau(patience=5, verbose=1, monitor='val_loss')]
WEIGHTS_FILE = 'brats'
model.fit(x=x_train, y=y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test), callbacks=callbacks)  #
model.save_weights(WEIGHTS_FILE)
print(model.evaluate(x_test, y_test))

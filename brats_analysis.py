import cv2
import os
import numpy as np
from matplotlib import image
import models
import matplotlib.pyplot as plt
import nibabel


def load_data():
  train = []
  truth = []
  path = 'data/LGG/'
  dir = sorted(os.listdir(path))
  for folder in dir:
    fdir = sorted(os.listdir(path + folder))
    for file in fdir:
      x = nibabel.load(path + folder + '/' + file)
      data = x.get_fdata()
      if 'seg' in file:
        truth.append(data)
      else:
        train.append(data)
  return train, truth


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
def readin():
  pass



x,y = load_data()
print(len(x))
print(len(y))
x = np.asarray(x)
y = np.asarray(y)
print(x.shape)
print(y.shape)
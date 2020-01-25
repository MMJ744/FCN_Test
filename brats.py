import cv2
import os
import numpy as np
from matplotlib import image
import nibabel

def load_data():
  train = []
  truth = []
  path = 'data/brats_test/'
  dir = sorted(os.listdir(path))
  for file in dir:
    x = nibabel.load(path + file)
    train.append(x)
    print(x.shape)
load_data()
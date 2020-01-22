from test import load_data, unet_model, dice_coef
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from matplotlib import image
import os
from collections import Counter
def display_all(predicts):
    for p in predicts:
        cutoff = 0.5
        p[p > cutoff] = 1
        p[p <= cutoff] = 0
        p = cv2.merge((p, p, p))
        plt.imshow(p, interpolation='nearest')
        plt.show()

def correctly_predicted(zipped):
    accuracy = []
    for t, p in zipped:
        accuracy.append(np.sum(t == p))
    return accuracy

def f1(zipped):
    f1 = []
    pre = []
    rec = []
    for t,p in zipped:
        totalP = np.sum(t)
        totalPPredicted = np.sum(p)
        tp = np.sum(t * p)
        p = tp / totalPPredicted
        r = tp / totalP
        f = 2*p*r / (p+r)
        if not math.isnan(f):
            pre.append(p)
            rec.append(r)
            f1.append(f)
    return f1, pre, rec

def showInfo(arr):
    print('max  ' + str(max(arr)))
    print('avg  ' + str(sum(arr) / len(arr)))
    print('min  ' + str(min(arr)))
    print('> 0.75   ' + str(sum(i > 0.75 for i in arr)))
    print('< 0.25   ' + str(sum(i < 0.25 for i in arr)))
    print('---')

def eval(truth, predicted):
    predicted = np.rint(predicted)
    zipped = zip(truth,predicted)
    f,p,r = f1(zipped)
    showInfo(f)

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
models = [unet_model(),unet_model(),unet_model(),unet_model(),unet_model()]
weight_base = 'newdata32-'
for i in range(5):
    models[i].load_weights(weight_base+str(i+1)+'.h5')
x=200
given = test[x]
given = given.reshape((1,) + given.shape)
correct = test_truth[x]
#predictedWhole = models[4].predict(test)
#predictedWhole = np.rint(predictedWhole)
#print("------")
#zipped = zip(truth,predictedWhole)
#accuracy = correctly_predicted(zipped)
#zipped = zip(truth,predictedWhole)
#f1, pre, rec = f1(zipped)
#showInfo(f1)
#showInfo(pre)
#showInfo(rec)
#print("------")

#plt.imshow(correct, interpolation='nearest')
#plt.show()
cutoff = 0.3
predicts = []
for model in models:
    #print(model.evaluate(test,truth))
    output = model.predict(test)
    #output[output >= cutoff] = 1
    #output[output < cutoff] = 0
    predicts.append(output)

average = (predicts[0] + predicts[1] + predicts[2] + predicts[3] + predicts[4]) / 5 # Weighted voting

#Get dice of them

for p in predicts:
    p[p >= cutoff] = 1
    p[p < cutoff] = 0
    #print(dice_coef(correct,p))
print(np.max(average))
cutoff = 0.3
average[average >= cutoff] = 1.0
average[average < cutoff] = 0.0

print(np.max(average))
#print(dice_coef(correct,average))
show = False
if (show):
    average = cv2.merge((average, average, average))
    plt.imshow(average, interpolation='nearest')
    plt.show()
    display_all(predicts)

for i in range(5):
    print("Model " + str(i))
    eval(test_truth, predicts[i])
print("Ensamble")
eval(test_truth, average)



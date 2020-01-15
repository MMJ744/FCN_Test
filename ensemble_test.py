from test import load_data, unet_model, dice_coef
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
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

input, truth = load_data()
input = np.asarray(input)
truth = np.asarray(truth)
truth = truth.reshape(truth.shape + (1,))
models = [unet_model(),unet_model(),unet_model(),unet_model(),unet_model()]
weight_base = 'test32-200-'
for i in range(5):
    models[i].load_weights(weight_base+str(i+1)+'.h5')
x=200
given = input[x]
given = given.reshape((1,) + given.shape)
correct = truth[x]
#predictedWhole = models[4].predict(input)
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
predicts = []
for model in models:
    #print(model.evaluate(input,truth))
    output = model.predict(input)
    predicts.append(output)

average = (predicts[0] + predicts[1] + predicts[2] + predicts[3] + predicts[4]) / 5.0
#Get dice of them
cutoff = 0.5
for p in predicts:
    p[p > cutoff] = 1
    p[p <= cutoff] = 0
    #print(dice_coef(correct,p))
average[average > cutoff] = 1
average[average <= cutoff] = 0
#print(dice_coef(correct,average))
show = False
if (show):
    average = cv2.merge((average, average, average))
    plt.imshow(average, interpolation='nearest')
    plt.show()
    display_all(predicts)

for i in range(5):
    print("Model " + str(i))
    eval(truth, predicts[i])
print("Ensamble")
eval(truth, average)



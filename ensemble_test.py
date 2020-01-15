from test import load_data, unet_model, dice_coef
import numpy as np
import cv2
import matplotlib.pyplot as plt

def display_all(predicts):
    for p in predicts:
        cutoff = 0.5
        p[p > cutoff] = 1
        p[p <= cutoff] = 0
        p = cv2.merge((p, p, p))
        plt.imshow(p, interpolation='nearest')
        plt.show()

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
#plt.imshow(correct, interpolation='nearest')
#plt.show()
predicts = []
for model in models:
    print(model.evaluate(input,truth))
    output = model.predict(given)[0]
    predicts.append(output)

average = (predicts[0] + predicts[1] + predicts[2] + predicts[3] + predicts[4]) / 5.0
#Get dice of them
cutoff = 0.5
for p in predicts:
    print(p.shape)
    print(correct.shape)
    p[p > cutoff] = 1
    p[p <= cutoff] = 0
    #print(dice_coef(correct,p))
average[average > cutoff] = 1
average[average <= cutoff] = 0
#print(dice_coef(correct,average))


average = cv2.merge((average, average, average))
plt.imshow(average, interpolation='nearest')
plt.show()



display_all(predicts)

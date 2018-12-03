import numpy as np
import os
from PIL import Image

import sys
import cv2

path = os.getcwd() + '/data/X_train_task7.npy'
X_tr = np.load(path)
path = os.getcwd() + '/data/y_train_task7.npy'
y_tr = np.load(path)
print('X_tr shape: ', X_tr.shape)

for i in range(10):
    im = X_tr[i]
    #im = cv2.resize(X_tr[i], (32,32))
    im = Image.fromarray(im)
    if y_tr[i] == 0:
        y = 'nodamage'
    elif y_tr[i] == 1:
        y = 'minor'
    elif y_tr[i] == 2:
        y = 'moderate'
    elif y_tr[i] == 3:
        y = 'heavy'
    im.save('imgs/' + str(i) + '_' + y + '.png')
    #im.save('imgs/resized_' + str(i) + '_' + y + '.png')



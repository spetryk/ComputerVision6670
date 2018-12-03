import numpy as np
import cv2
from PIL import Image

X = np.load('./data/X_train_task7.npy')
y = np.load('./data/y_train_task7.npy')

#X = X.astype(dtype='float64')
#y = y.astype(dtype='float64')

val_ratio = 0.9

s = np.arange(X.shape[0])
np.random.seed(42)
np.random.shuffle(s)
shuffleX = X[s]
shuffley = y[s]

idx = int(X.shape[0]*0.9)

X_tr = X[:idx, :, :, :]
y_tr = y[:idx]
X_val = X[idx:, :, :, :]
y_val = y[idx:]
"""
np.save('./data/X_tr_task7.npy', X_tr)
np.save('./data/y_tr_task7.npy', y_tr)
np.save('./data/X_val_task7.npy', X_val)
np.save('./data/y_val_task7.npy', y_val)

print(X.shape)

r_mean = []
g_mean = []
b_mean = []

r_std = []
g_std = []
b_std = []

for img in X:
    r_mean.append(np.mean(img[:,:,0]))
    g_mean.append(np.mean(img[:,:,1]))
    b_mean.append(np.mean(img[:,:,2]))

    r_std.append(np.std(img[:,:,0]))
    g_std.append(np.std(img[:,:,1]))
    b_std.append(np.std(img[:,:,2]))

print('means: ', np.mean(r_mean), np.mean(g_mean), np.mean(b_mean))
print('stds:  ', np.std(r_std), np.std(g_std), np.std(b_std))
"""

for i, img in enumerate(X_tr):
    im = Image.fromarray(img)
    im.save('./data/train/' + str(y_tr[i]) + '/' + str(i) + '.png')

for i, img in enumerate(X_val):
    im = Image.fromarray(img)
    im.save('./data/val/' + str(y_val[i])+ '/' + str(i) + '.png')






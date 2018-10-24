import numpy as np

def shuffle(images, labels):
    # shuffle pairs of data
    n_y = len(labels)
    indices = np.arange(n_y)
    np.random.shuffle(indices)
    sx, sy = [], []
    for idx, val in enumerate(indices):
        sx.append(images[val,:,:,:])
        sy.append(labels[val])
    sx, sy = np.array(sx), np.array(sy)
    return sx, sy

def split(images, labels, split_ratio):
    # split into training and validation sets
    n_y = len(labels)
    split_index = np.round(split_ratio*n_y)
    X_train =
    X_valid =

def load_data(split_ratio):
    # load data from np files
    images = np.load('../phidata/X_train.npy')
    labels = np.load('../phidata/y_train.npy')
    # split into test/train
    simages, slabels = shuffle(images, labels)
    split(images, labels, split_ratio)


load_data(0.2)

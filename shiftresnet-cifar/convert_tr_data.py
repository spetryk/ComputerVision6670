import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.onnx

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import h5py
import numpy as np
from torch.utils.data import DataLoader
from utils import utils
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.autograd import Variable

#Import models
from models.resnet import ResNet20
import time

use_cuda = torch.cuda.is_available()
print("use_cuda"+str(use_cuda))
# best_acc = 0.0  # best test accuracy


### GDefine configurations


TRAIN_DATA_DIR='./data/train/'
TRAIN_FILE=TRAIN_DATA_DIR+'train_set.h5'

TEST_DATA_DIR='./data/test/'
TEST_FILE=TEST_DATA_DIR+'test_set.h5'

### Read Training Data
print("===> Loading datasets")
#train_set = utils.H5Dataset(TRAIN_FILE)
#trainloader = DataLoader(dataset=train_set, shuffle=True,  batch_size=300 , num_workers=2)


print('Reading train dataset')
train_velocities, train_images = utils.load_data(TRAIN_FILE)
print(train_images.shape)
print(train_images.dtype)
print(train_velocities.shape)
print(train_velocities.dtype)

train_images = train_images.astype(dtype ='float64')
train_velocities = train_velocities.astype(dtype ='float64')

np.save('./data/train/train_velocities.npy', train_velocities)
np.save('./data/train/train_images.npy', train_images)


# train_set=utils.Dataset(train_images, train_velocities)


#Validation set
test_velocities, test_images = utils.load_data(TEST_FILE)
test_images = test_images.astype(dtype ='float64')
test_velocities = test_velocities.astype(dtype ='float64')
# test_set=utils.Dataset(test_images, test_velocities)

np.save('./data/test/test_velocities.npy', test_velocities)
np.save('./data/test/test_images.npy', test_images)

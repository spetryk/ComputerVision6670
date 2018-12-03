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

import numpy as np
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from collections import OrderedDict
from torch.autograd import Variable

from utils import progress_bar

#Import models
from models.resnet import ResNet20
import time

use_cuda = torch.cuda.is_available()
print("use_cuda"+str(use_cuda))
best_acc = 0.0  # best test accuracy


### GDefine configurations


print("===> Preparing data")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((127.855, 120.010, 114.571), (17.584, 17.378, 17.786)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((127.855, 120.010, 114.571), (17.584, 17.378, 17.786)),
])

### Read Training Data
print("===> Loading datasets")
train_images = np.load('./data/X_tr_task7.npy')
train_labels = np.load('./data/y_tr_task7.npy')


#Validation set
test_images = np.load('./data/X_val_task7.npy')
test_labels = np.load('./data/y_val_task7.npy')

#test_set=utils.Dataset(test_images, test_labels)

# Datasets
trainset = torchvision.datasets.ImageFolder('./data/train', transform_train)
testset = torchvision.datasets.ImageFolder('./data/val', transform_test)

num_samples = train_images.shape[0]
num_samples_test = test_images.shape[0]
batch_size=128
total_epochs=1
learning_rate=0.1
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

trainloader = DataLoader(dataset=trainset, shuffle=True,  batch_size=batch_size , num_workers=8)
testloader = DataLoader(dataset=testset, shuffle=True,  batch_size=batch_size , num_workers=8)


########### Model
print("===> Building model")
MODEL_DIR='./trained_models/'
MODEL_NAME=MODEL_DIR+'Resnet20'
net = ResNet20(num_classes=4)
#backup original net - to save model later
original_net = ResNet20(num_classes=4)

criterion =  nn.CrossEntropyLoss()

#combine above two
#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel


###########



### Use GPUs for training
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


def adjust_learning_rate(epoch, lr):
    if epoch <= 12:  # 32k iterations
      return lr
    elif epoch <= 18:  # 48k iterations
      return lr/10
    else:
      print('Reducing lr at epoch ', epoch)
    return lr/100

ce_loss=[]
ce_epoch_loss=[]
def train(epoch):
    lr = adjust_learning_rate(epoch, learning_rate)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    epoch_loss=[]
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * targets.size(0)

        print("-----------------------------------------------------")
        print("training Batch ID: ", batch_idx)
        # print("Training Batch Loss: ", loss.data.cpu().numpy()[0])  #py 3
        print("Training Batch Loss: ", loss.item()) #py2 gives tensor
        print("-----------------------------------------------------")


        epoch_loss.append(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (train_loss / total, 100. * float(correct) / float(total), correct, total))
    ce_loss.append([epoch_loss])

    print("########################################################")
    print("training Epoch: ", epoch)

    e= np.array(epoch_loss)
    avg_epoch_loss = ( e.sum() * batch_size) / num_samples
    ce_epoch_loss.append(avg_epoch_loss)
    print("training avg_epoch_loss : ", avg_epoch_loss)
    print("########################################################")

test_ce_loss = []
test_ce_epoch_loss = []

best_loss = 100000


path='./checkpoint/'

def validate(epoch):

    global best_loss
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    t_epoch_loss=[]
    for batch_idx, (inputs, targets) in enumerate(testloader):
        #with torch.no_grad():
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        print("-----------------------------------------------------")
        print("test Batch ID: ", batch_idx)
        print("Test Batch Loss: ", loss.item())
        print("-----------------------------------------------------")
        t_epoch_loss.append(loss.item())
        test_loss += loss.item() * targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / total, 100. * float(correct) / float(total), correct, total))
    test_ce_loss.append([t_epoch_loss]);

    print("########################################################")
    print("test Epoch: ", epoch)

    e = np.array(t_epoch_loss)
    avg_epoch_loss = (e.sum() * batch_size) / num_samples_test
    test_ce_epoch_loss.append(avg_epoch_loss)
    print("test avg_epoch_loss : ", avg_epoch_loss)
    print("########################################################")

    # Save checkpoint.
    acc = 100. * float(correct) / float(total)

    model_name = MODEL_NAME+'_loss_'+str(round(avg_epoch_loss,2))+'_epoch_'+str(epoch)+'.pth'


    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, model_name)
        print('* Saved checkpoint to %s' % path)
        best_acc = acc


start_time = time.time()
for epoch in range(start_epoch, total_epochs):
    train(epoch)
    validate(epoch)

end_time = time.time()

print("******************")
print("Time:  ",end_time-start_time)
print(ce_epoch_loss)
print(test_ce_epoch_loss)

#####Save the model
#Ref https://github.com/onnx/tutorials/blob/master/tutorials/PytorchOnnxExport.ipynb

#
m_output = MODEL_NAME+'_1.pth'
m_output_onnx = MODEL_NAME+'_1.onnx'
# torch.save(net.state_dict(), 'dp_'+m_output)


# state_dict = torch.load(MODEL_NAME)
state_dict = net.state_dict()


new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
original_net.load_state_dict(new_state_dict)

# torch.save(original_net.state_dict(), 'state_dict_'+m_output)

torch.save(original_net,m_output)





dummy_input = Variable(torch.randn(1, 3, 48, 96))
torch.onnx.export(original_net, dummy_input, m_output_onnx)

#### Save the loss
np.save(MODEL_DIR+'ce_epoch_loss.npy', np.array(ce_epoch_loss))
np.save(MODEL_DIR+'test_ce_epoch_loss.npy', np.array(test_ce_epoch_loss))


###Frames per second

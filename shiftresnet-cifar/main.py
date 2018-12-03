'''Train CIFAR10 with PyTorch.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import ResNet20
from models import ResNet56
from models import ResNet110
#from models import ShiftResNet20
#from models import ShiftResNet56
#from models import ShiftResNet110
from models import DepthwiseResNet20
from models import DepthwiseResNet56
from models import DepthwiseResNet110
from utils import progress_bar
from torch.autograd import Variable


all_models = {
    'resnet20': ResNet20,
    #'shiftresnet20': ShiftResNet20,
    'depthwiseresnet20': DepthwiseResNet20,
    'resnet56': ResNet56,
    #'shiftresnet56': ShiftResNet56,
    'depthwiseresnet56': DepthwiseResNet56,
    'resnet110': ResNet110,
    #'shiftresnet110': ShiftResNet110,
    'depthwiseresnet110': DepthwiseResNet110
}

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size', '-b', default=128, type=int, help='batch size')
parser.add_argument('--arch', '-a', choices=all_models.keys(), default='resnet20', help='neural network architecture')
parser.add_argument('--expansion', '-e', help='Expansion for shift resnet.', default=1, type=float)
parser.add_argument('--reduction', help='Amount to reduce raw resnet model by', default=1.0, type=float)
parser.add_argument('--reduction-mode', choices=('block', 'net', 'depthwise'), help='"block" reduces inner representation for BasicBlock, "net" reduces for all layers', default='net')
parser.add_argument('--dataset', choices=('cifar10', 'cifar100', 'imagenet'), help='Dataset to train and validate on.', default='cifar100')
parser.add_argument('--datadir', help='Folder containing data', default='./data/')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    num_classes=10
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    num_classes = 100
elif args.dataset == 'imagenet':
    raise NotImplementedError()
    transform_train = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std = [ 0.229, 0.224, 0.225 ]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std = [ 0.229, 0.224, 0.225 ]),
    ])

    traindir = os.path.join(args.datadir, 'train')
    valdir = os.path.join(args.datadir, 'val')
    trainset = torchvision.datasets.ImageFolder(traindir, transform_train)
    testset = torchvision.datasets.ImageFolder(valdir, transform_test)
    num_classes = 1000
elif args.dataset == 'task7':
    traindir = os.path.join(args.datadir, 'train')
    valdir = os.path.join(args.datadir, 'val')
    trainset = torchvision.datasets.ImageFolder(traindir, transform_train)
    testset = torchvision.datasets.ImageFolder(valdir, transform_test)
    num_classes = 4

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)


if 'shift' in args.arch:
    suffix = '_%s' % args.expansion
elif args.reduction != 1:
    suffix = '_%s_%s' % (args.reduction, args.reduction_mode)
else:
    suffix = ''

if args.dataset == 'cifar100':
    suffix += '_cifar100'

if args.dataset == 'imagenet':
    suffix += '_imagenet'

path = './checkpoint/%s%s.t7' % (args.arch, suffix)
print('Using path: %s' % path)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint.. %s' % path)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(path)
    net = checkpoint['net']
    best_acc = float(checkpoint['acc'])
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    cls = all_models[args.arch]
    assert 'shift' not in args.arch or args.reduction == 1, \
        'Only default resnet and depthwise resnet support reductions'
    if args.reduction != 1:
        print('==> %s with reduction %.2f' % (args.arch, args.reduction))
        net = cls(reduction=args.reduction, reduction_mode=args.reduction_mode, num_classes=num_classes)
    else:
        net = cls(args.expansion, num_classes=num_classes) if 'shift' in args.arch else cls(num_classes=num_classes)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def adjust_learning_rate(epoch, lr):
    if epoch <= 81:  # 32k iterations
      return lr
    elif epoch <= 122:  # 48k iterations
      return lr/10
    else:
      print('Reducing lr at epoch ', epoch)
      return lr/100

# Training
def train(epoch):
    lr = adjust_learning_rate(epoch, args.lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
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
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/total, 100.*float(correct)/float(total), correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/total, 100.*float(correct)/float(total), correct, total))

    # Save checkpoint.
    acc = 100.*float(correct)/float(total)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, path)
        print('* Saved checkpoint to %s' % path)
        best_acc = acc


for epoch in range(start_epoch, 1):
    train(epoch)
    test(epoch)

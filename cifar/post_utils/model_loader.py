'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
import numpy as np

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import util


import sys
sys.path.append("../")
sys.path.append("../..")
# import utils_gsp.gpu_projection as gsp_gpu
# import utils_gsp.padded_gsp as gsp_model
# import utils_gsp.sps_tools as sps_tools

def load_model(model_path, model_type):
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    # Use CUDA
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')

    best_acc = 0  # best test accuracy



    # Data
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
    # if args.dataset == 'cifar10':
    dataloader = datasets.CIFAR10
    num_classes = 10
    # else:
    #     dataloader = datasets.CIFAR100
    #     num_classes = 100


    trainset = dataloader(root='../data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers= 4)

    testset = dataloader(root='../data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers= 4)

    # Model
    print("==> creating model '{}'".format('resnet'))
    model = models.__dict__['resnet'](
                num_classes=num_classes,
                depth=56,
                block_name='BasicBlock',
            )

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    if model_type == 'baseline':
        # Loading Basline Models
        title = 'cifar-10-' + 'resnet'
        # model_path = 'cifar/results/resnet-56_gsp0.99_/2020-11-06_03-05-12/model.pth'
        checkpoint = torch.load(os.path.join('', './checkpoints/cifar10/resnet-56-baseline/model_best'+'.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
    
    elif model_type == 'sparse-model':
        ## Loading Sparsified Models
        title = 'cifar-10-' + 'resnet'
        # model_path = './results/resnet-56_global_gsp-all_0.8_/11-16_20-20/model.pth'
        model.load_state_dict(torch.load(model_path))

    return model
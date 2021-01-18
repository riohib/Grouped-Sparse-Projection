import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import sys

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from datetime import datetime

import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import util

#os.makedirs('saves', exist_ok=True)
warnings.filterwarnings("ignore")


import sys
sys.path.append("../")
sys.path.append("../..")
import utils_gsp.gpu_projection as gsp_gpu
import utils_gsp.padded_gsp as gsp_model
import utils_gsp.sps_tools as sps_tools


# ===================================================================================

# ===================================================================================

class Args:
    data = '/data/users2/rohib/github/imagenet-data'
    arch = 'resnet50'
    reg = 0
    decay = 4e-5
    workers = 16
    epochs = 5
    start_epoch = 0
    batch_size = 256
    lr = 0.001
    lr_decay = 0.1
    lr_int = 30
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 100
    resume = False
    evaluate = False
    pretrained = ''
    world_size = 1
    dist_url = 'tcp://224.66.41.62:23456'
    dist_backend = 'gloo'
    seed = None
    gpu = None
    sensitivity = 1e-4
    sps = 0.9

# ===================================================================================
# ===================================================================================
def load_model_pytorch(model_path, model_type, args):
    
    global best_acc1
    args.gpu = 0

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    
    model.cuda()

    return model

# ===================================================================================


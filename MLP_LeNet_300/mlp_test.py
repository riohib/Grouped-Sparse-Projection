import argparse
import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from net.models import LeNet
import util
import pdb

import time
import scipy.io

from matplotlib import pyplot as plt
import logging

import sys
sys.path.append("../")
import utils_gsp.vec_projection as gsp_vec
import utils_gsp.var_projection as gsp_reg
import utils_gsp.gpu_projection as gsp_gpu
import utils_gsp.padded_gsp as gsp_model
import utils_gsp.sps_tools as sps_tools




# Select Device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

# Load Model

model = LeNet(mask=False).to(device)
gspm_path = './saves/S0.97/0.97_0_pre.pth'

model.load_state_dict(torch.load(gspm_path,  map_location=device)) 


gsp_param_dict = {}
sps_list = []
ind = 0
for name, param in model.named_parameters():
    if 'weight' in name:
        gsp_param_dict[ind] = param.detach()
        sps_list.append( sps_tools.sparsity(gsp_param_dict[ind]))
        ind += 1

print(sps_list)



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

import time
import scipy.io
import copy

from matplotlib import pyplot as plt
import logging

from net.models import LeNet_5 as LeNet
import util

from utils.helper import *
import utils.sps_tools as sps_tools

import utils.vec_projection as gsp_vec
import utils.var_projection as gsp_reg
import utils.gpu_projection as gsp_gpu
import utils.padded_gsp as gsp_model

# Select Device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

# Load Model

model_dh = LeNet(mask=False).to(device)
model_gsp = LeNet(mask=False).to(device)
dhm_path = './saves/DH/elt_0.01_2.pth'
gspm_path = './saves/S0.97/S0.97_3_pre.pth'

model_dh.load_state_dict(torch.load(dhm_path))
model_gsp.load_state_dict(torch.load(gspm_path)) 

dh_param_dict = {}
for name, p in model_dh.named_parameters():
    dh_param_dict[name] = torch.flatten(p)

gsp_param_dict = {}
for name, p in model_dh.named_parameters():
    gsp_param_dict[name] = torch.flatten(p)


# ------------------- Test Sparsity

def print_sparsity(model_dh):
    w1 = model_dh.conv1.weight.detach()
    w2 = model_dh.conv2.weight.detach()
    w3 = model_dh.fc1.weight.detach()
    w4 = model_dh.fc2.weight.detach()

    reshaped_w1 = w1.view(20,25)
    reshaped_w2 = w2.view(250, 100)
    reshaped_w3 = w3
    reshaped_w4 = w4

    print("Layer 1 Sparsity w1: %.2f \n" % (gsp_vec.sparsity(reshaped_w1)))
    print("Layer 2 Sparsity w2: %.2f \n" % (gsp_vec.sparsity(reshaped_w2)))
    print("Layer 3 Sparsity w3: %.2f \n" % (gsp_vec.sparsity(reshaped_w3)))
    print("Layer 3 Sparsity w4: %.2f \n" % (gsp_vec.sparsity(reshaped_w4)))


def save_mat(model):
    w1 = model.conv1.weight.detach()
    w2 = model.conv2.weight.detach()
    w3 = model.fc1.weight.detach()
    w4 = model.fc2.weight.detach()

    reshaped_w1 = w1.view(20,25).numpy()
    reshaped_w2 = w2.view(250, 100).numpy()
    reshaped_w3 = w3.view(50, -1).numpy()
    reshaped_w4 = w4.view(50, -1).numpy()

    mdict = {'arr1': reshaped_w1.reshape(-1), 'arr2': reshaped_w2.reshape(-1), \
             'arr3': reshaped_w3.reshape(-1), 'arr4': reshaped_w4.reshape(-1)   } 

    scipy.io.savemat('matrix_cell.mat', mdict=mdict, long_field_names=True)

    # scipy.io.savemat('matrix_1.mat', mdict={'arr': reshaped_w1.reshape(-1)})
    # scipy.io.savemat('matrix_2.mat', mdict={'arr': reshaped_w2.reshape(-1)})
    # scipy.io.savemat('matrix_3.mat', mdict={'arr': reshaped_w3.reshape(-1)})
    # scipy.io.savemat('matrix_4.mat', mdict={'arr': reshaped_w4.reshape(-1)})

    with open('matrix_1.pkl', 'wb') as handle:
        pickle.dump(reshaped_w1, handle)

    with open('matrix_2.pkl', 'wb') as handle:
        pickle.dump(reshaped_w2, handle)

    with open('matrix_3.pkl', 'wb') as handle:
        pickle.dump(reshaped_w3, handle)

    with open('matrix_4.pkl', 'wb') as handle:
        pickle.dump(reshaped_w4, handle)

print_sparsity(model_dh)
print_sparsity(model_gsp)


dh_param_dict['conv1.weight']

dh_weights = {
    'conv1' : dh_param_dict['conv1.weight'],
    'conv2' : dh_param_dict['conv2.weight'],
    'fc1' : dh_param_dict['fc1.weight'],
    'fc2' : dh_param_dict['fc2.weight'],
}


gsp_weights = {
    'conv1' : gsp_param_dict['conv1.weight'],
    'conv2' : gsp_param_dict['conv2.weight'],
    'fc1' : gsp_param_dict['fc1.weight'],
    'fc2' : gsp_param_dict['fc2.weight'],
}


## Global Model Projection

model = copy.deepcopy(model_dh)
sps_tools.cnn_model_sps(model)

in_dict = sps_tools.cnn_make_dict(model)

gsp_model.sparsity_dict(in_dict)
X, ni_list = gsp_model.groupedsparseproj(in_dict, 0.9)
out_dict = gsp_model.unpad_output_mat(X, ni_list)
gsp_model.sparsity_dict(out_dict)


# Put Dict back into model
sps_tools.cnn_dict_to_model(model, out_dict)

import logging
#************************* Imports *************************

import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from numpy import linalg as LA

from matplotlib import pyplot as plt
import numpy as np
import pickle

import utils.torch_projection as gsp_reg
import utils.vec_projection as gsp_vec

np.random.seed(0)
torch.manual_seed(0)

#************************* Imports *************************

logging.basicConfig(filename='LogFile.log', level=logging.DEBUG)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

list = []
nan_list = []
nan_iter = []

def sort(a):
    sortIdx = np.argsort(LA.norm(a, 1, axis=1))
    q = np.reshape(a[sortIdx[0], :], (1, a.shape[1]))

    for i in range( a.shape[0] - 1, -1, -1 ):
        ind = sortIdx[i]
        r = np.reshape(a[ind, :], (1, a.shape[1]))
        q = np.concatenate((q, r), axis=0)
    return q


def weight_splot(version, model):
    fig = plt.figure(1)
    weights = model.e1.weight.detach().numpy()
    weightr =sort(weights)
    c = 0
    for i in range(128):
        plt.subplot(12, 12, c + 1)
        plt.imshow( torch.tensor(weightr[c]).view(28, 28), cmap=plt.cm.RdBu_r)
        plt.clim(-1, 1)
        c += 1
        plt.axis('off')
    plt.savefig('weight_graph_{}.png'.format(version))


# def weight_plot(iter):
#     plt.figure(2)
#     weights = model.e1.weight.detach()
#     io = to_img(weights)
#     plt.imshow(io[2].view(28, 28))
#     save_image(io, './weights_plot/wp_{}.png'.format(iter))


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def LeNet300_sps(model):
    pDict = {} 
    for name, pA in model.named_parameters(): 
        pDict.update({name : pA})
    
    p_size = [] 
    for items in paramList: 
        p_size.append(items.size()) 

def parameter_prune(model, threshold = 2.5e-2):
    l1 = model.fc1.weight 
    l2 = model.fc2.weight 
    l3 = model.fc3.weight

    numel1 = (l1 < threshold).nonzero()
    numel2 = (l2 < threshold).nonzero()
    numel3 = (l3 < threshold).nonzero()

    pruned1 = numel1.shape[0]
    pruned2 = numel2.shape[0]
    pruned3 = numel3.shape[0]

    left1 = l1.numel() - pruned1
    left2 = l2.numel() - pruned2
    left3 = l3.numel() - pruned3

    pct1 = left1/l1.numel()
    pct2 = left2/l2.numel()
    pct3 = left3/l3.numel()
    
    print('Parameter Left Layer 1: ' + str(left1) + ' Percentage: ' + str(pct1))
    print('Parameter Left from Layer 2: ' + str(left2) + ' Percentage: ' + str(pct2))
    print('Parameter Left from Layer 3: ' + str(left3) + ' Percentage: ' + str(pct3))

def model_prune(model, threshold = 2.3e-2):
    # for name, p in model.named_parameters():
    #     if 'mask' in name:
    #         continue
    pDict = {} 
    for name, pA in model.named_parameters(): 
        pDict.update({name : pA})
    
    keys = ['fc1.weight','fc2.weight','fc3.weight']
    
    for k in keys:
        print(k)
        tensor = pDict[k].data.cpu().numpy()
        tensor = np.where(tensor < threshold, 0, tensor)
        pDict[k].data = torch.from_numpy(tensor).to(device)

def fix_zeros(model):
    # for name, p in model.named_parameters():
    #     if 'mask' in name:
    #         continue

    pDict = {} 
    for name, pA in model.named_parameters(): 
        pDict.update({name : pA})
    keys = ['fc1.weight','fc2.weight','fc3.weight']

    for k in keys:
        tensor = pDict[k].data.cpu().numpy()
        grad_tensor = pDict[k].grad.data.cpu().numpy()
        grad_tensor = np.where(tensor==0, 0, grad_tensor)
        pDict[k].grad.data = torch.from_numpy(grad_tensor).to(device)
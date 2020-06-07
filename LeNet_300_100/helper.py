# from projection import sparse_opt
from gs_projection import *

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

np.random.seed(0)
torch.manual_seed(0)

#************************* Imports *************************

logging.basicConfig(filename='LogFile.log', level=logging.DEBUG)


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



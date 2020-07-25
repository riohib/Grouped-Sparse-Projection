import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statistics


from net.models import LeNet_5 as LeNet
import util

import utils.vec_projection as gsp_vec



def layer_wise_sps(model):
    w1 = model.conv1.weight.detach()
    w2 = model.conv2.weight.detach()
    w3 = model.fc1.weight.detach()
    w4 = model.fc2.weight.detach()

    reshaped_w1 = w1.view(20,25)
    reshaped_w2 = w2.view(250, 100)
    reshaped_w3 = w3
    reshaped_w4 = w4

    print("Layer 1 Sparsity w1: %.2f \n" % (gsp_vec.sparsity(reshaped_w1)))
    print("Layer 2 Sparsity w2: %.2f \n" % (gsp_vec.sparsity(reshaped_w2)))
    print("Layer 3 Sparsity w3: %.2f \n" % (gsp_vec.sparsity(reshaped_w3)))
    print("Layer 4 Sparsity w4: %.2f \n" % (gsp_vec.sparsity(reshaped_w4)))


def model_sps(model):
    w1 = model.conv1.weight.detach()
    w2 = model.conv2.weight.detach()
    w3 = model.fc1.weight.detach()
    w4 = model.fc2.weight.detach()

    reshaped_w1 = w1.view(500,-1)
    reshaped_w2 = w2.view(500,-1)
    reshaped_w3 = w3.view(500,-1)
    reshaped_w4 = w4.view(500,-1)
    
    tot_weight = torch.cat([reshaped_w1,reshaped_w2,reshaped_w3,reshaped_w4], dim=1)
    print("Total Model Sparsity w1: %.2f \n" % (gsp_vec.sparsity(tot_weight)))


def cnn_layer_Ploter(model, title):
    subRow = 4
    subCol = 5
    c = 0
    plt.figure(figsize=(15,10))
    fig, axes = plt.subplots(subRow, subCol)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for i in range(20):
        plt.subplot(subRow, subCol, c + 1)
        im = plt.imshow(model.conv1.weight[i].detach().view(5,5), cmap=plt.cm.RdBu_r)
        c+=1

    fig.suptitle(title)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
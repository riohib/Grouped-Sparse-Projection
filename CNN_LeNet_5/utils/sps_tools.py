import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import logging

from math import gcd
from functools import reduce
from operator import mul

from net.models import LeNet_5 as LeNet
import util
import utils.vec_projection as gsp_vec
# import utils.torch_projection as gsp_reg
import utils.var_projection as gsp_reg
import utils.gpu_projection as gsp_gpu

logging.basicConfig(filename = 'logElem.log' , level=logging.DEBUG)


#=====================================================================================================
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
    # print("Total Model Sparsity w1: %.2f \n" % (gsp_vec.sparsity(tot_weight)))
    return gsp_vec.sparsity(tot_weight)

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


#=====================================================================================================
## Select which GSP Function to use:
gs_projection = gsp_reg.groupedsparseproj
# gs_projection = gsp_vec.groupedsparseproj
# gs_projection = gsp_gpu.groupedsparseproj
#=====================================================================================================

def global_gsp(model, itr, sps):
    params_d = {}
    weight_d = {}
    for name, param in model.named_parameters(): 
        params_d[name] = param.detach()
        if 'weight' in name:
            weight_d[name] = param.detach()
    
    # Calculate the row_size of the input matrix for GSP. Using the GCD of the product 
    # of the tensor dimensions as the row_size and set the required column size for each
    # layer weight matrix.
    shape_list = [reduce(mul, list(y.shape)) for x, y in weight_d.items()]   
    gcd_dim = reduce(gcd, shape_list)
    #------------------------------------------------------------------------

    global_w_tup = ()
    second_dim = []

    for k, v in weight_d.items():
        reshaped = v.view(int(gcd_dim)/10, -1)
        second_dim.append(reshaped.shape[1])  # store second dim for putting back tensor into model.
        global_w_tup = global_w_tup + (reshaped,) #creating new tuple, as torch.cat takes tuple.

    global_weights = torch.cat(global_w_tup, dim=1)

    sparse_g_weights = gs_projection(global_weights, sps)

    i = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            
            if i == 0:
                start = 0
                end = second_dim[i]
            else:
                start = second_dim[i-1]
                end = second_dim[i-1] + second_dim[i]
            param.data = sparse_g_weights[:, start:end].clone().requires_grad_(True).view(param.shape)
            
            i += 1
    
    if itr % 600 == 0:
        logging.debug(f" ------------------- itr No: {itr} ------------------ \n")
        logging.debug(f" Global Model Sparsity: {model_sps(model)} \n")



#=====================================================================================================
#=====================================================================================================
def gsp(model, itr, sps):

    w1 = model.conv1.weight.detach()
    w2 = model.conv2.weight.detach()
    w3 = model.fc1.weight.detach()
    w4 = model.fc2.weight.detach()

    reshaped_w1 = w1.view(20,25)
    reshaped_w2 = w2.view(250, 100)
    reshaped_w3 = w3
    reshaped_w4 = w4

    sparse_w1 = gs_projection(reshaped_w1, 0.91)
    sparse_w2 = gs_projection(reshaped_w2, 0.85)
    sparse_w3 = gs_projection(reshaped_w3, 0.72)
    sparse_w4 = gs_projection(reshaped_w4, 0.33)
    
    model.conv1.weight.data = sparse_w1.clone().requires_grad_(True).view(20,1,5,5)
    model.conv2.weight.data = sparse_w2.clone().requires_grad_(True).view(50,20,5,5)
    model.fc1.weight.data = sparse_w3.clone().requires_grad_(True)
    model.fc2.weight.data = sparse_w4.clone().requires_grad_(True)

    if itr % 600 == 0:
        logging.debug(" ------------------- itr No: %s ------------------ \n" % itr)
        logging.debug("Layer 1 Sparsity w1 | before: %.2f | After: %.2f \n" % 
                        (gsp_vec.sparsity(reshaped_w1), gsp_vec.sparsity(model.conv1.weight.detach().view(20,25))))
        logging.debug("Layer 2 Sparsity w2 | before: %.2f | After: %.2f \n" % 
                        (gsp_vec.sparsity(reshaped_w2), gsp_vec.sparsity(model.conv2.weight.detach().view(250,100))))
        logging.debug("Layer 3 Sparsity w3 | before: %.2f | After: %.2f \n" % 
                        (gsp_vec.sparsity(reshaped_w3), gsp_vec.sparsity(model.fc1.weight.detach())))
        logging.debug("Layer 3 Sparsity w3 | before: %.2f | After: %.2f \n" % 
                        (gsp_vec.sparsity(reshaped_w4), gsp_vec.sparsity(model.fc2.weight.detach())))
# ===================================== GSP FUNCTION ===========================================

def var_GSP(model, itr, sps):
    weight_d = {}
    shape_list = []

    counter = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            shape_list.append(param.data.shape)
            weight_d[counter] = param.detach().view(-1)
            counter += 1
    
    sps_weight = gs_projection(weight_d, sps)
    
    counter = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            shape = shape_list[counter]
            param.data = sps_weight[counter].view(shape).requires_grad_(True)
            counter += 1


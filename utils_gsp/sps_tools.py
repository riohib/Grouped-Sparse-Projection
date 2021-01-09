import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import logging
import pickle

from math import gcd
from functools import reduce
from operator import mul

# from net.models import LeNet_5 as LeNet
# import util

import sys
sys.path.append('../')
import utils_gsp.padded_gsp as gsp_global
import utils_gsp.gpu_projection as gsp_gpu


logging.basicConfig(filename = 'logElem.log' , level=logging.DEBUG)

# Select Device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

#======================================================================================================
#====================================== Sparsity Calculator ===========================================
#=====================================================================================================
def sparsity(matrix):
    ni = matrix.shape[0]

    # Get Indices of all zero vector columns.
    zero_col_ind = (abs(matrix.sum(0) - 0) < 2.22e-16).nonzero().view(-1)  
    spx_c = (np.sqrt(ni) - torch.norm(matrix,1, dim=0) / torch.norm(matrix,2, dim=0)) / (np.sqrt(ni) - 1)
    if len(zero_col_ind) != 0:
        spx_c[zero_col_ind] = 1  # Sparsity = 1 if column already zero vector.
    
    if matrix.dim() > 1:   
        sps_avg =  spx_c.sum() / matrix.shape[1]
    elif matrix.dim() == 1:  # If not a matrix but a column vector!
        sps_avg =  spx_c    
    return sps_avg

def sparsity_dict(in_dict):
    r = len(in_dict)
    spx = 0
    spxList = []
    for i in range(r):
        if in_dict[i].sum() == 0:
            spx = 1
            spxList.append(spx)
        else:
            ni = in_dict[i].shape[0]
            spx = (np.sqrt(ni) - torch.norm(in_dict[i], 1) / torch.norm(in_dict[i], 2)) / (np.sqrt(ni) - 1)
            spxList.append(spx)
        spx = sum(spxList) / r
    return spx
  
def model_weight_sps(model):
    gsp_param_dict = {}
    sps_list = []
    ind = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            gsp_param_dict[ind] = param.detach()
            sps_list.append(sparsity(gsp_param_dict[ind]))
            ind += 1
    print(sps_list)


#=====================================================================================================
#=====================================================================================================

def apply_gsp(model, sps, gsp_func = gsp_gpu):
    """
    This function is for applying GSP layer-wise in a CNN or MLP or Resnet network in this repo.
    The GSP is applied layer-wise separately.
    """
    weight_d = {}
    shape_list = []
    counter = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            shape_list.append(param.data.shape)    
            if param.dim() > 2:  #Only two different dimension for LeNet-5
                w_shape = param.shape
                weight_d[counter] = param.detach().view(50,-1)
                param.data = gsp_func.groupedsparseproj(weight_d[counter], sps).view(w_shape)
            else:
                param.data = gsp_func.groupedsparseproj(param.detach(), sps)
            counter += 1


def get_layerwise_sps(model):
    counter = 0
    weight_d = {}
    sps_list = []
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            # shape_list.append(param.data.shape)
            if param.dim() > 2:  #Only two different dimension for LeNet-5
                weight_d[counter] = param.detach().view(50,-1)
                sps_list.append(sparsity(weight_d[counter])) 
            else:
                sps_list.append(sparsity(param.data)) 
            counter += 1

    print(f"Layerwise sps: L1: {sps_list[0]:.4f} | L2: {sps_list[1]:.4f} | \
        L3: {sps_list[2]:.4f} | L4: {sps_list[3]:.4f}")
    return sps_list

def get_cnn_layer_shape(model):
    counter = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            print(f"Paramater Shape: {param.shape}")
            counter += 0

# def cnn_model_sps(model):
#     w1 = model.conv1.weight.detach()
#     w2 = model.conv2.weight.detach()
#     w3 = model.fc1.weight.detach()
#     w4 = model.fc2.weight.detach()

#     reshaped_w1 = w1.view(500,-1)
#     reshaped_w2 = w2.view(500,-1)
#     reshaped_w3 = w3.view(500,-1)
#     reshaped_w4 = w4.view(500,-1)
    
#     tot_weight = torch.cat([reshaped_w1,reshaped_w2,reshaped_w3,reshaped_w4], dim=1)
#     print("Total Model Sparsity w1: %.2f \n" % (sparsity(tot_weight)))
#     return sparsity(tot_weight)

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

## ====================================================================== ##
## =========== Helper Functions for Global GSP with pad ================= ##
## ====================================================================== ##
def make_weight_dict(model, arch):
    in_dict = {}
    counter = 0
    if arch == 'cnn':
        for name, param in model.named_parameters(): 
            if 'weight' in name:
                in_dict[counter] = param.detach().view(-1)
                counter += 1

    elif arch == 'resnet':
        for name, param in model.named_parameters(): 
            if 'weight' in name and 'module.conv1' not in name and 'bn' not in name and 'downsample' \
                not in name and 'fc' not in name:
                in_dict[counter] = param.detach().view(-1)
                counter += 1
    elif arch == 'resnet-not-bn':
        for name, param in model.named_parameters(): 
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                in_dict[counter] = param.detach().view(-1)
                counter += 1
    return in_dict


def dict_to_model(model, out_dict, arch):
    param_d = {}
    index = 0

    if arch == 'cnn':
        for name, param in model.named_parameters(): 
            if 'weight' in name:
                layer_shape = param.shape
                param_d[index] = param
                # print(layer_shape)
                # print(f"out-shape: {out_dict[index].shape}")
                param.data = out_dict[index].view(layer_shape)
                index += 1
    elif arch == 'resnet':
        for name, param in model.named_parameters(): 
            if 'weight' in name and 'module.conv1' not in name and 'bn' not in name and 'downsample' \
                not in name and 'fc' not in name:
                layer_shape = param.shape
                param_d[index] = param
                param.data = out_dict[index].view(layer_shape)
                index += 1
    elif arch == 'resnet-not-bn':
        for name, param in model.named_parameters(): 
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                layer_shape = param.shape
                param_d[index] = param
                param.data = out_dict[index].view(layer_shape)
                index += 1

## ====================================================================== ##
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


# ===================================== GSP FUNCTION ===========================================
def gsp_global_apply(model, sps, arch):
    ## Global Model Projection
    in_dict = make_weight_dict(model, arch)
    try:
        X, ni_list = gsp_global.groupedsparseproj(in_dict, sps)
    except:
        # import pdb; pdb.set_trace()
        print(gsp_global.groupedsparseproj(in_dict, sps))
        with open('problem_mat.pickle', 'wb') as handle:
            pickle.dump(in_dict, handle)

    out_dict = gsp_global.unpad_output_mat(X, ni_list)

    # Put Dict back into model
    dict_to_model(model, out_dict, arch)




## ============================================================================ ##
## ================================ GSP-Resnet ================================ ##
## ============================================================================ ##

def gsp_resnet_partial(model, sps=0.95, gsp_func = gsp_gpu):
    """
    This function is for applying GSP layer-wise in a ResNet network in this repo.
    The GSP is applied layer-wise separately.  
    """
    params_d = {}
    weight_d = {}
    shape_list = []
    counter = 0

    for name, param in model.named_parameters(): 
        params_d[name] = param
        if 'weight' in name and 'module.conv1' not in name and 'bn' not in name and 'downsample' not in name and 'fc' not in name:
            shape_list.append(param.data.shape)
            weight_d[name] = param  
            w_shape = param.shape
            dim_1 = w_shape[0] * w_shape[1]
            weight_d[counter] = param.detach().view(dim_1,-1)
            param.data = gsp_func.groupedsparseproj(weight_d[counter], sps).view(w_shape)


# def gsp_resnet_all_layers(model, sps=0.95, gsp_func = gsp_gpu):
#     """
#     This function is for applying GSP layer-wise in a ResNet network in this repo.
#     The GSP is applied layer-wise separately.  
#     """
#     params_d = {}
#     weight_d = {}
#     shape_list = []
#     counter = 0
#     for name, param in model.named_parameters(): 
#         params_d[name] = param

#         if 'weight' in name and 'module.conv1' not in name and 'downsample' not in name and 'fc' not in name:
#             shape_list.append(param.data.shape)
#             weight_d[name] = param  

#             w_shape = param.shape
#             weight_d[counter] = param.detach().view(256,-1)
#             param.data = gsp_func.groupedsparseproj(weight_d[counter], sps).view(w_shape)


def resnet_layerwise_sps(model):
    counter = 0
    weight_d = {}
    sps_dict = {}
    shape_dict = {}
    for name, param in model.named_parameters(): 
        if 'weight' in name and 'bn' not in name and 'downsample' not in name:
        # if 'weight' in name:
            shape_dict[name] = param.detach().shape            
            weight_d[counter] = param.detach().view(16,-1)
            sps_dict[name] = sparsity(weight_d[counter]).item()
    return sps_dict, shape_dict


def get_model_methods(obj):
    obj_methods = [method_name for method_name in dir(obj) if callable(getattr(obj, method_name))]
    for elem in obj_methods:
        print(elem)

def resnet_dict_weights(model):
    params_d = {}
    weight_d = {}
    shape_list = {}
    counter = 0
    bn_total = 0
    for name, param in model.named_parameters(): 
        params_d[name] = param
        # if 'weight' in name and 'module.conv1' not in name and 'downsample' not in name and 'fc' not in name:
        if 'weight' in name and 'bn' in name:
            shape_list[name] = param.data.shape
            weight_d[name] = param
            bn_total += param.data.shape[0]

    return params_d, weight_d




## ============================================================================ ##
## =============================== GSP-Imagenet =============================== ##
## ============================================================================ ##

def gsp_imagenet_partial(model, sps=0.95, gsp_func = gsp_gpu):
    """
    This function is for applying GSP layer-wise in a ResNet network for the imagenet dataset.
    The GSP is applied layer-wise separately.  
    """
    params_d = {}
    weight_d = {}
    shape_list = []
    counter = 0

    for name, param in model.named_parameters(): 
        params_d[name] = param
        if 'weight' in name and 'module.conv1' not in name and 'bn' not in name and 'downsample' not in name and 'fc' not in name:
            shape_list.append(param.data.shape)
            weight_d[name] = param  
            w_shape = param.shape
            dim_1 = w_shape[0] * w_shape[1]
            weight_d[counter] = param.detach().view(dim_1,-1)
            param.data = gsp_func.groupedsparseproj(weight_d[counter], sps).view(w_shape)
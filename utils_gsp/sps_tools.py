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
import os

# logging.basicConfig(filename = 'logElem.log' , level=logging.DEBUG)

# Select Device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

## ================================================================================================= ##
# --------------------------------------------- LOGGER ---------------------------------------------- #
def create_logger(log_dir, logfile_name, if_stream = True):
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # ----------------------------------------------------------------------
    logger_name = logfile_name

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:     :%(message)s')

    logger_path = log_dir + logger_name + '.log'

    file_handler = logging.FileHandler(logger_path)
    
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if if_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def print_model_parameters(model, logger, with_values=False):
    logger.info(f"{'Param name':20} {'Shape':30} {'Type':15}")
    logger.info('-'*70)
    for name, param in model.named_parameters():
        logger.info(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            logger.info(param)


#======================================================================================================
#====================================== Sparsity Calculator ===========================================
#=====================================================================================================
def sparsity(matrix):
    ni = matrix.shape[0]

    # Get Indices of columns with all-0 vectors.
    zero_col_ind = (abs(matrix.sum(0) - 0) < 2.22e-16).nonzero().view(-1)  
    spx_c = (np.sqrt(ni) - torch.norm(matrix,1, dim=0) / torch.norm(matrix,2, dim=0)) / (np.sqrt(ni) - 1)
    if len(zero_col_ind) != 0:
        spx_c[zero_col_ind] = 1  # Sparsity = 1 if column already zero vector.
    
    if matrix.dim() > 1:   
        # sps_avg =  spx_c.sum() / matrix.shape[1]
        sps_avg = spx_c.mean()
    elif matrix.dim() == 1:  # If not a matrix but a column vector!
        sps_avg =  spx_c    
    return sps_avg

def padded_sparsity(matrix, ni_list):
    """
    This Hoyer Sparsity Calculation is for matrices with the end of the columns that are padded. Hence,
    it needs the information of how much of each columns are elements and how much of them are padded.
    ni_list: Contains the number of values in each column (rest are padded with zero).
    """

    ni = matrix.shape[0]
    ni_tensor = torch.tensor(ni_list).to(device)

    # Get Indices of all zero vector columns.
    zero_col_ind = (abs(matrix.sum(0) - 0) < 2.22e-16).nonzero().view(-1)  
    spx_c = (torch.sqrt(ni_tensor) - torch.norm(matrix,1, dim=0) / torch.norm(matrix,2, dim=0)) / (torch.sqrt(ni_tensor) - 1)
    if len(zero_col_ind) != 0:
        spx_c[zero_col_ind] = 1  # Sparsity = 1 if column already zero vector.
    
    if matrix.dim() > 1:   
        sps_avg = spx_c.mean()
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
                dim_1 = w_shape[0] * w_shape[1]
                weight_d[counter] = param.detach().view(dim_1,-1)
                param.data = gsp_func.groupedsparseproj(weight_d[counter], sps).view(w_shape)
            else:
                param.data = gsp_func.groupedsparseproj(param.detach(), sps)
            counter += 1

def get_layerwise_sps(model):
    """
    This function will output a list of layerwise sparsity of a LeNet-5 CNN model.
    The sparsity measure is the Hoyer Sparsity Measure.
    """
    counter = 0
    weight_d = {}
    sps_d = {}
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            if param.dim() > 2:  #Only two different dimension for LeNet-5
                w_shape = param.shape
                dim_1 = w_shape[0] * w_shape[1]
                weight_d[counter] = param.detach().view(dim_1,-1)
                sps_d[name] = sparsity(weight_d[counter])
            else:
                sps_d[name] = sparsity(param.data)
            counter += 1
    
    w_name_list = [x for x in sps_d.keys()] 

    return sps_d

def get_cnn_layer_shape(model):
    counter = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            print(f"Paramater Shape: {param.shape}")
            counter += 0

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

## ================================================================================================ ##
## ======================== Helper Functions for Global GSP with pad ============================== ##
## ================================================================================================ ##
# ----------------------------------------------------------------------------------------------------
# These set of Functions are for concatenating all the NN Layer weights side by side and using GSP. 
# These differ from the next set of functions which flatten the layers and stack them side by side 
# (utilizing dict data structures). These functions however simply create a tensor and passes to
# padded GSP.

def apply_concat_gsp(model, sps):
    matrix, val_mask, shape_l = concat_nnlayers(model)
    try:
        type(matrix) == torch.Tensor
    except:
        print("The output of concat_nnlayers - 'matrix' is not a Torch Tensor!")

    xp_mat, ni_list = gsp_global.groupedsparseproj(matrix, val_mask, sps)
    rebuild_nnlayers(xp_mat, ni_list, shape_l, model)

    return xp_mat, ni_list

def concat_nnlayers(model):
    shape_l = []
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:
            shape_l.append(tuple(param.shape)) 
    
    max_dim0 = sum([x[0] for x in shape_l])
    max_dim1 = sum([x[1] for x in shape_l])
    
    matrix = torch.zeros(max_dim0, max_dim1, device=device)
    val_mask = torch.zeros(max_dim0, max_dim1, device=device)
    
    counter = 0
    dim1 = 0
    ni_list = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            prev_dim0, cur_dim0 = shape_l[counter-1][0], shape_l[counter][0]
            prev_dim1, cur_dim1 = shape_l[counter-1][1], shape_l[counter][1]

            matrix[0:cur_dim0, dim1:dim1+cur_dim1] = param.data
            val_mask[0:cur_dim0, dim1:dim1+cur_dim1] = torch.ones(shape_l[counter]).to(device)
            
            dim1 += cur_dim1
            counter += 1
    return matrix, val_mask, shape_l

def rebuild_nnlayers(matrix, ni_list, shape_l, model):
    counter = 0
    dim1 = 0
    ni_list = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            prev_dim0, cur_dim0 = shape_l[counter-1][0], shape_l[counter][0]
            prev_dim1, cur_dim1 = shape_l[counter-1][1], shape_l[counter][1]
            param.data = matrix[0:cur_dim0, dim1:dim1+cur_dim1]

            dim1 += cur_dim1
            counter += 1

### ===================================================================================================
### ===================================================================================================

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
        # if 'weight' in name and 'bn' not in name and 'downsample' not in name:
        if 'weight' in name:
            if param.dim() > 2:
                shape_dict[name] = param.detach().shape   
                w_shape = param.shape
                dim_1 = w_shape[0] * w_shape[1]         
                weight_d[counter] = param.detach().view(dim_1,-1)
                sps_dict[name] = sparsity(weight_d[counter]).item()
            else:
                sps_dict[name] = sparsity(param.data).item()

    # Average Sparsity of the whole Model:
    sps_list = [vals for vals in sps_dict.values()]
    avg_sps = sum(sps_list)/len(sps_list)

    return avg_sps, sps_dict, shape_dict


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

def get_model_layers(model):
    params_d = {}
    for name, param in model.named_parameters(): 
        params_d[name] = param
    
    layer_list = [x for x in params_d.keys()]
    return layer_list

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
        if 'weight' in name and 'layer' in name and 'bn' not in name and 'downsample' not in name: # and 'fc' not in name
            shape_list.append(param.data.shape)
            weight_d[name] = param  
            w_shape = param.shape
            dim_1 = w_shape[0] * w_shape[1]
            weight_d[counter] = param.detach().view(dim_1,-1)
            param.data = gsp_func.groupedsparseproj(weight_d[counter], sps).view(w_shape)
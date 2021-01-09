import sys
sys.path.append("../..")
import utils_gsp.gpu_projection as gsp_gpu
import utils_gsp.padded_gsp as gsp_model
import utils_gsp.sps_tools as sps_tools

import numpy as np
import matplotlib.pyplot as plt

def weight_sparsity_dist(model, model_name='resnet-56'):
    
    ## Get Layerwise Sparsity
    sps_dict, shape_dict = sps_tools.resnet_layerwise_sps(model)

    sps_list = []
    layer_list = []
    for key, val in sps_dict.items():
        layer_list.append(key)
        sps_list.append(val)

    # Plotting
    plt.style.use('seaborn')
    model_name = 'resnet'

    # fig1, ax1 = plt.subplots()
    # ax1.bar = (layer_list, sps_list)

    fig = plt.figure(figsize=[20,10])

    ax = fig.add_axes([0,0,1,1])
    ax.bar(layer_list, sps_list)

    plt.xticks(rotation='vertical')
    plt.yticks(rotation='vertical')
    
    plt.title(model_name + ' layer-wise sparsity')
    plt.xlabel('Layers')
    plt.ylabel('Sparsity')
    plt.tight_layout()
    plt.savefig(model_name + 'layerwise_sps.png')

import sys
sys.path.append("../..")
import utils_gsp.gpu_projection as gsp_gpu
import utils_gsp.padded_gsp as gsp_model
import utils_gsp.sps_tools as sps_tools

import model_loader
import gsp_plotter as gplot


model_path = '../results/resnet-56_global_gsp-all_0.8_/11-16_20-20/model.pth'
model = model_loader.load_model(model_path, 'sparse-model')

## Get Layerwise Sparsity
spl, shp = sps_tools.resnet_layerwise_sps(model)

## Layer SPS Bar Plot
gplot.weight_sparsity_dist(model)
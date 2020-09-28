import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch.optim as optim

from matplotlib import pyplot as plt

# from convNet import *

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_classes = 10

# Create Directories
if not os.path.exists('./filter_images'):
    os.mkdir('./filter_images')
if not os.path.exists('./feature_maps'):
    os.mkdir('./feature_maps')

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.fc = nn.Linear(16*16*64, num_classes)
        
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = out3.reshape(out3.size(0), -1)
        out = self.fc(out)
        return (out1, out2, out3, out)

model = ConvNet(num_classes).to(device)
model_van = ConvNet(num_classes).to(device)
model_relu = ConvNet(num_classes).to(device)
model_gsp = ConvNet(num_classes).to(device)

model.load_state_dict(torch.load('convCeleba.pth'))
model_relu.load_state_dict(torch.load('convCelebaRelu.pth'))
# model_van.load_state_dict(torch.load('convVanilla.pth'))
# model_relu.load_state_dict(torch.load('convRelu.pth'))
# model_gsp.load_state_dict(torch.load('convNetDeep.pth'))


def weight_splot(model, filterShape ,layerNo):
    fig = plt.figure(1)

    if layerNo == 0:
        loopVar = 16
        subRow = 4
        subCol = 4
        weightr = model.layer1[0].weight.detach()
        channels = model.layer1[0].weight.shape[1]

    elif layerNo == 1:
        loopVar = 32
        subRow = 6
        subCol = 6
        weightr = model.layer2[0].weight.detach()
        channels = model.layer2[0].weight.shape[1]
    elif layerNo == 2:
        loopVar = 64
        subRow = 8
        subCol = 8
        weightr = model.layer3[0].weight.detach()
        channels = model.layer3[0].weight.shape[1]

    for n in range(channels):
        c = 0
        for i in range(loopVar):
            plt.subplot(subRow, subCol, c + 1)
            plt.imshow(weightr[c][n].view(filterShape, filterShape), cmap=plt.cm.RdBu_r)
            plt.clim(-1, 1)
            c += 1
            plt.axis('off')
        fig.suptitle("Filters_layer_7{} Channel_{}".format(layerNo+1, n))
        plt.savefig('Filters_L{}_Channel_7_{}.png'.format(layerNo, n))

fshape = 5
weight_splot(model, fshape, layerNo=0)
weight_splot(model, fshape, layerNo=1)
weight_splot(model, fshape, layerNo=2)

#=====================================================================================================

# Plot all the Filter Outputs and Feature Maps
def plot_FeatureMap(model, layerNo):
    
    if layerNo == 0:
        loopVar = 16
        subRow = 4
        subCol = 4
    elif layerNo == 1:
        loopVar = 32
        subRow = 6
        subCol = 6
    elif layerNo == 2:
        loopVar = 64
        subRow = 8
        subCol = 8

    # ones = torch.ones(64,3,128,128)/2 

    fig = plt.figure(4)
    for images, labels in test_loader: 
        images = images.to(device) 
        labels = labels.to(device) 
        outputs = model(images) 
        break 

    # outputs = model(ones)    

    c = 0
    for i in range(loopVar):
        plt.subplot(subRow, subCol, c + 1)
        # plt.imshow(weightr[c][0].view(5, 5), cmap=plt.cm.RdBu_r)
        plt.imshow(outputs[layerNo][0][i].detach().numpy())
        # plt.clim(-1, 1)
        c += 1
        plt.axis('off')
    
    fig.suptitle("Feature_Maps_Layer{} Filters".format(layerNo+1))
    plt.savefig('feature_maps/Feature_Map_L{}.png'.format(layerNo))

modelF = model
plot_FeatureMap(modelF, layerNo=0)
plot_FeatureMap(modelF, layerNo=1)
plot_FeatureMap(modelF, layerNo=2)



#=====================================================================================================
for images, labels in test_loader: 
    images = images.to(device) 
    labels = labels.to(device) 
    outputs = model(images) 
    break 

plt.imshow(outputs[0][0][0].detach().numpy()) 
plt.savefig('Layer1_output.png') 

plt.imshow(outputs[1][0][0].detach().numpy()) 
plt.savefig('Layer2_output.png') 

plt.imshow(outputs[2][0][0].detach().numpy()) 
plt.savefig('Layer3_output.png') 

plt.imshow(outputs[3][0][0].detach().numpy()) 
plt.savefig('Layer4_output.png') 


# Original Image Plotting
img = images[0]
img = img.detach().numpy().swapaxes(0,2) 
plt.imshow(imgg)
plt.savefig('Original_image.png')
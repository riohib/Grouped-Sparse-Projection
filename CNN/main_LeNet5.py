import os
from gs_projection import *
import logging

import numpy
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
num_epochs = 20
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)
model_relu = ConvNet(num_classes).to(device)
model_gsp = ConvNet(num_classes).to(device)

def gsp(model, itr, rgb=False):
    sps = 0.95

    sps = 0.8
    if rgb == True:
        mult = 3
    else:
        mult = 1

    w1 = model.layer1[0].weight.detach().numpy()
    w2 = model.layer2[0].weight.detach().numpy()
    # w3 = model.layer3[0].weight.detach().numpy()

    reshaped_w1 = numpy.reshape(w1, (16,25)) 
    reshaped_w2 = numpy.reshape(w2, (512,25))
    # reshaped_w3 = numpy.reshape(w3, (2048,25))  

    sparse_w1 = groupedsparseproj(reshaped_w1, sps, itr)
    sparse_w2 = groupedsparseproj(reshaped_w2, sps, itr)
    # sparse_w3 = groupedsparseproj(reshaped_w3, sps, itr)

    w_reshaped1 = numpy.reshape(sparse_w1, (16,1,5,5)) 
    w_reshaped2 = numpy.reshape(sparse_w2, (32,16,5,5))
    # w_reshaped3 = numpy.reshape(sparse_w3, (64,32,5,5))  

    model.layer1[0].weight.data = torch.tensor(w_reshaped1, dtype=torch.float32)
    model.layer2[0].weight.data = torch.tensor(w_reshaped2, dtype=torch.float32)
    # model.layer3[0].weight.data = torch.tensor(w_reshaped3, dtype=torch.float32)
    

    # if itr % 10 == 0:
    #     logging.debug(" ------------------- itr No: %s ------------------ \n" % itr)
    #     logging.debug("Layer 1 Sparsity w1 | before: %.2f | After: %.2f \n" % 
    #                             (sparsity(w1), sparsity(model.e1.weight.detach().numpy())))
    # print("---------------------------------------------------")


def relu_weights(model):
    w1 = model.layer1[0].weight.detach()
    w_pos1 = torch.nn.functional.relu(w1, inplace=False).numpy()
    cw1 = torch.tensor(w_pos1, dtype=torch.float32, requires_grad=True)
    model.layer1[0].weight.data = cw1

    w2 = model.layer2[0].weight.detach()
    w_pos2 = torch.nn.functional.relu(w2, inplace=False).numpy()
    cw2 = torch.tensor(w_pos2, dtype=torch.float32, requires_grad=True)
    model.layer2[0].weight.data = cw2

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
itr = 0
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Function Application
        if itr % 20 == 0:
            # relu_weights(model)
            gsp(model, itr, rgb=False)
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        itr += 1

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.pth')
# torch.save(model.state_dict(), './convRelu.pth')
torch.save(model.state_dict(), './convGSP.pth')

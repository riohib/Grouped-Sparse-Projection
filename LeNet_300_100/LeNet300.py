import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import scipy.io

from matplotlib import pyplot as plt
import logging

from utils.helper import *

import utils.vec_projection as gsp_vec
import utils.torch_projection as gsp_reg

from net.models import LeNet


## New Post Conf
filepath = './results/E3_VecGsp90_e100_test/'

#******************** Result Directories **************************
if not os.path.exists('./Loss'):
    os.mkdir('./Loss')
if not os.path.exists('./weights_plot'):
    os.mkdir('./weights_plot')

if not os.path.exists('./results'):
    os.mkdir('./results')

if not os.path.exists(filepath):
    os.mkdir(filepath)

# --------------------------- Logging ------------------------------------------
logging.basicConfig(filename = filepath + 'log_file_test.log', level=logging.DEBUG)

# ---------------------- Device configuration ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========================== GSP FUNCTION ==================================
sps = 0.9

## Select which GSP Function to use:
# gs_projection = gsp_reg.groupedsparseproj
#---------------------------------------------------------------------------
gs_projection = gsp_vec.groupedsparseproj

def gsp(model, itr, trial_list, sps = 0.9):
    w1 = model.fc1.weight.detach()
    w2 = model.fc2.weight.detach()
    w3 = model.fc3.weight.detach()

    # print(itr)
    # if (itr == 40):
    #     scipy.io.savemat('w1.mat', mdict={'w1': w1})
    #     scipy.io.savemat('w2.mat', mdict={'w2': w2})
    #     scipy.io.savemat('w3.mat', mdict={'w3': w3})

    sparse_w1 = gs_projection(w1, sps)
    sparse_w2 = gs_projection(w2, sps)
    sparse_w3 = gs_projection(w3, sps)

    model.fc1.weight.data = sparse_w1.clone().requires_grad_(True)
    model.fc2.weight.data = sparse_w2.clone().requires_grad_(True)
    model.fc3.weight.data = sparse_w3.clone().requires_grad_(True)

    
    if itr < 600:
        trial_list.append(sparse_w1.sum().item())
        trial_list.append(sparse_w2.sum().item())
        trial_list.append(sparse_w3.sum().item())

    if itr % 10 == 0:
        logging.debug(" ------------------- itr No: %s ------------------ \n" % itr)
        logging.debug("Layer 1 Sparsity w1 | before: %.2f | After: %.2f \n" % 
                        (gsp_vec.sparsity(w1), gsp_vec.sparsity(model.fc1.weight.detach())))
        logging.debug("Layer 2 Sparsity w2 | before: %.2f | After: %.2f \n" % 
                        (gsp_vec.sparsity(w2), gsp_vec.sparsity(model.fc2.weight.detach())))
    return trial_list
# ========================== GSP FUNCTION ==================================


# Hyper-parameters 
input_size = 784
hidden_size1 = 300
hidden_size2 = 100
num_classes = 10

num_epochs = 100
gsp_interval = 10

batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                        train=True, 
                                        transform=transforms.ToTensor(),  
                                        download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                        train=False, 
                                        transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=False)


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()   # Runs the __init__ of the parent class.

#         self.e1 = nn.Linear(28*28, 300)  # flattened image dim is the input = 28*28, output=64
#         self.e2 = nn.Linear(300, 100)
#         self.e3 = nn.Linear(100, 10)
    
#     def forward(self, x):
#         x = F.relu(self.e1(x))
#         x = F.relu(self.e2(x))
#         x = self.e3(x)
#         return F.log_softmax(x, dim=1)


# model_us = Net().to(device)

def train(num_epochs):
    # Select Model Class
    model = LeNet(mask=False).to(device)

    ## Load Pretrained Model
    model_filepath = './PreTrained/'
    model.load_state_dict(torch.load(model_filepath + 'LeNet300_pt.pth', map_location=device))

    # Loss and optimizer
    critrion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


    # Prune Model
    # model_prune(model, threshold=2.0e-2)
    # Train the model
    loss_array = []
    trial_list = []
    gsp_interval = 20
    in_sparse = 0
    itr = 0
    total_step = len(train_loader)
    model.train()

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            # if itr % gsp_interval == 0:
            #     trial_list = gsp(model, itr, trial_list, sps)
            #     print("GSP-ing: itr:  " + str(itr))

            # Forward pass
            outputs = model(images)
            loss = critrion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            if itr % gsp_interval == 0:
                trial_list = gsp(model, itr, trial_list, sps)
                print("GSP-ing: itr:  " + str(itr))

            # Fix Zeros Mask
            # fix_zeros(model)

            optimizer.step()
            loss_array.append(loss)

            if (i+1) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            itr += 1

    return model, trial_list

# filepath = './gsp_models_trained/'
# model.load_state_dict(torch.load(filepath + 'LeNet_PREgsp_98_T_0.005_e200_T_0.008.pth', map_location='cpu'))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

def test(model):
    epoch = 100
    gsp_interval = 10
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    
    start_time = time.time()
    model, trial_list = train(num_epochs)
    print("--- %s seconds ---" % (time.time() - start_time))

torch.save(model.state_dict(), filepath + './ln_GspVec_Post' + str(sps) + '_ep' \
                            + str(num_epochs) + '_i' + str(gsp_interval) +'.pth')
test(model)


    # fpp = './results/E1_vgsp90_e200/'
    # model.load_state_dict(torch.load(fpp + 'LN_Pre_0.9_ep100_i20.pth', map_location='cpu'))
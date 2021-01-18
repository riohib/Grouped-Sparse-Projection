import argparse
import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

import time
import scipy.io
from matplotlib import pyplot as plt
import logging


from net.models import LeNet_5 as LeNet
import util

import sys
sys.path.append("../")
import utils_gsp.vec_projection as gsp_vec
import utils_gsp.var_projection as gsp_reg
import utils_gsp.gpu_projection as gsp_gpu
import utils_gsp.padded_gsp as gsp_model
import utils_gsp.sps_tools as sps_tools


os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of epochs to train (default: 100)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-step', type=int, default=80, metavar='LR-STEP',
                    help='learning rate scheduler step (default: 75)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=12345678, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')

parser.add_argument('--sps', type=float, default=0.95, metavar='SPS',
                    help='gsp sparsity value (default: 0.95)')
parser.add_argument('--gsp-int', type=int, default=50, metavar='GSP-INT',
                    help='gsp sparsity value (default: 50)')  
parser.add_argument('--gsp-pre-stop', type=int, default=0, metavar='GSP-INT',
                    help='gsp sparsity value (default: 50)')  
parser.add_argument('--device', type=str, default='gpu',
                    help='cpu or gpu gsp version')     

parser.add_argument('--pretrained', type=str, default='./saves/elt_0.0_0',
                    help='the path to the pretrained model')
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

# Loader
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

#==============================================================================================
logging.basicConfig(filename = 'logElem.log' , level=logging.DEBUG)

gsp_interval = 20; sps=args.sps

# Define which model to use
model = LeNet(mask=False).to(device)

print(model)
util.print_model_parameters(model)

# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=args.lr)
initial_optimizer_state_dict = optimizer.state_dict()

def train(epochs, threshold=0.0):
    itr=0
    model.train()
    pbar = tqdm(range(epochs), total=epochs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    curves = np.zeros((epochs,14))
    
    for epoch in pbar:
        # print(f" learning rate: {optimizer.param_groups[0]['lr']}")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = F.nll_loss(output, target)
                  
            total_loss = loss
            total_loss.backward()
            optimizer.step()

            # Projection using GSP
            if (itr % args.gsp_int == 0) and epoch <= (args.epochs - args.gsp_pre_stop):
                sps_tools.apply_gsp(model, args.sps, gsp_func = gsp_gpu)
                last_gsp_itr = itr
            
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f"Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} \
                ({percentage:3.0f}%)]  Loss: {loss.item():.3f}  LR: {optimizer.param_groups[0]['lr']:.4f} \
                GSP in itr: {last_gsp_itr} sps: {args.sps:.2f} Total Progress:" )
            itr+=1
        scheduler.step()


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} \
                    ({accuracy:.2f}%)')
    return accuracy

if args.pretrained:
    # model.load_state_dict(torch.load('saves/elt_0.0_0.pth'))
    # model.load_state_dict(torch.load('saves/F96_elt_0.0_0_'+str(args.gsp)+'.pth'))
    # model.load_state_dict(torch.load('saves/gsp_elt_0.0_0_'+str(args.gsp)+'.pth'))
    model.load_state_dict(torch.load(args.pretrained + '.pth'))
    accuracy = test()

# Initial training
print("--- Initial training ---")
train(args.epochs, threshold=0.0)
accuracy = test()
# torch.save(model.state_dict(), 'saves/S'+ str(args.sps)+'/S'+ str(args.sps)+'_3_'+str(args.gsp)+'.pth')
torch.save(model.state_dict(), 'saves/2021_gsp_stop/S'+ str(args.sps)+ '.pth')

util.log(args.log, f"initial_accuracy {accuracy}")
#util.print_nonzeros(model)
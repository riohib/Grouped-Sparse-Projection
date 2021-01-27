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
parser.add_argument('--log-dir', type=str, default='.logs/',
                    help='log file name')
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

parser.add_argument('--save-dir', type=str, default='./saves/',
                    help='the path to the model saved after training.')
parser.add_argument('--save-filename', type=str, default='gsp',
                    help='the path to the model saved after training.')  

parser.add_argument('--pretrained', type=str, default='./saves/elt_0.0_0',
                    help='the path to the pretrained model')
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# -------------------------- LOGGER ---------------------------------------------------- #
summary_logger = sps_tools.create_logger(args.log_dir, 'summary')
epoch_logger = sps_tools.create_logger(args.log_dir, 'training', if_stream = False)
# -------------------------------------------------------------------------------------- #


# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    summary_logger.info("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    summary_logger.info('Not using CUDA!!!')

# Generate arg values for printing with the report:
summary_logger.info(f"All the arguments used are:")
for arg in vars(args):
    summary_logger.info(f"{arg : <20}: {getattr(args, arg)}")
summary_logger.info("------------------------------------------------------------")


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
if args.lr_step > args.epochs:
    is_lro = 'no'
else:
    is_lro = 'yes'

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

save_path = args.save_dir + args.save_filename + '_'+str(args.gsp_int)+ '_seed_' + str(args.seed) + 'lro_' + str(is_lro) +'.pth'


#==============================================================================================
# Define which model to use
model = LeNet(mask=False).to(device)

# ------------ Log: Model Paramters and Shape to Summary -------------- #
summary_logger.info(model)
sps_tools.print_model_parameters(model, summary_logger)

# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=args.lr)
initial_optimizer_state_dict = optimizer.state_dict()

def train(epochs, threshold=0.0):
    best = 0.
    model.train()
    pbar = tqdm(range(epochs), total=epochs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    curves = np.zeros((epochs,14))
    accuracy_list = []
    
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
            if (batch_idx % args.gsp_int == 0) and epoch <= (args.epochs - args.gsp_pre_stop):
                sps_tools.apply_gsp(model, args.sps, gsp_func = gsp_gpu)
                last_gsp_itr = batch_idx
            
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f"Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} \
                ({percentage:3.0f}%)]  Loss: {loss.item():.3f}  LR: {optimizer.param_groups[0]['lr']:.4f} \
                GSP in itr: {last_gsp_itr} sps: {args.sps:.2f} Total Progress:" )

                epoch_logger.info(f"Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.3f}  LR: {optimizer.param_groups[0]['lr']:.4f} GSP in itr: {last_gsp_itr} sps: {args.sps:.2f} Total Progress:" )

        # Keep Track of best model
        accuracy, _ , _ = test()
        accuracy_list.append(accuracy)

        if accuracy > best:
            best = accuracy
            save_model_checkpoint(model, epoch, loss, PATH=save_path)

        scheduler.step()


# ===================================== TESTING ======================================= #
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
        epoch_logger.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy, test_loss, correct

def test_best(model_path, model):
    model = load_checkpoint(PATH=model_path)
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
        summary_logger.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy, test_loss, correct

def save_model_checkpoint(model, epoch, loss, PATH):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)

def load_checkpoint(PATH):
    model = LeNet(mask=False).to(device)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


# ===============================================================================================
if args.pretrained:
    model.load_state_dict(torch.load(args.pretrained + '.pth'))
    accuracy = test()

# Initial training
accuracy, _, _ = test()
summary_logger.info(f" Accuracy of Pretrained Model: {accuracy}")

summary_logger.info("--- Initial training ---")
train(args.epochs, threshold=0.0)

accuracy, _, _ = test_best(save_path, model)

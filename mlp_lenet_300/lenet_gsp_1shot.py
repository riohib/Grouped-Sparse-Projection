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

from net.models import LeNet

import sys
sys.path.append("../")
import utils_gsp.gpu_projection as gsp_gpu
import utils_gsp.sps_tools as sps_tools


os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 100)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')     
parser.add_argument('--lr-step', type=int, default=75, metavar='LR-STEP',
                    help='learning rate (default: 75)')     

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=12345678, metavar='S',
                    help='random seed (default: 42)')

parser.add_argument('--model', type=str, default='saves/initial_model',
                    help='path to model pretrained with sparsity-inducing regularizer')                    
parser.add_argument('--sensitivity', type=float, default=0.25,
                    help="pruning threshold computed as sensitivity value multiplies to layer's std")

parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log-dir', type=str, default='.logs/',
                    help='log file name')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')

parser.add_argument('--save-dir', type=str, default='./saves/',
                    help='the path to the model saved after training.')
parser.add_argument('--save-filename', type=str, default='gsp',
                    help='the path to the model saved after training.')  

parser.add_argument('--sps', type=float, default=0.97, metavar='SPS',
                    help='gsp sparsity value (default: 0.95)')
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
# -------------------------------------------------------------------------------------- #

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
save_path = args.save_dir + 'V_'+str(args.sensitivity)+'.pth'
#==============================================================================================


def train(epochs):
    best = 0.0
    accuracy_dict = {}

    pbar = tqdm(range(epochs), total=epochs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    for epoch in pbar:
        accuracy = test()
        if accuracy > best:
            best = accuracy
            torch.save(model.state_dict(), save_path)
            accuracy_dict[epoch] = best
            
        model.train()    
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            total_loss = loss
                
            total_loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Best: {best:.2f}% Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f} Total: {total_loss.item():.6f}')
                epoch_logger.info(f'Best: {best:.2f}% Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f} Total: {total_loss.item():.6f}')
        scheduler.step()
    
    summary_logger.info(f"Accuracy list of Best Epochs: {accuracy_dict}")
    summary_logger.info(f'Accuracy: {best:.2f}%')

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
        #print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

def load_checkpoint(PATH):
    model = LeNet(mask=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    initial_optimizer_state_dict = optimizer.state_dict()

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint)
    return model, optimizer

# ===================================================================================================

# Load the Model
model, optimizer = load_checkpoint(args.model+'.pth')
summary_logger.info(f"Accuracy of the loaded model: {test()}")


# Prune Using GSP
summary_logger.info(" ------------ Pruning with GSP ------------ ")
sps_tools.apply_gsp(model, args.sps, gsp_func = gsp_gpu)

# Initial training
summary_logger.info(" -------------- Pruning ---------------- ")
for name, p in model.named_parameters():
    if 'mask' in name:
        continue
    tensor = p.data.cpu().numpy()
    threshold = args.sensitivity*np.std(tensor)
    summary_logger.info(f'Pruning with threshold : {threshold} for layer {name}')
    new_mask = np.where(abs(tensor) < threshold, 0, tensor)
    p.data = torch.from_numpy(new_mask).to(device)

accuracy = test()
sps_tools.print_nonzeros(model, logger=summary_logger)

summary_logger.info("------------ Finetuning ------------")
summary_logger.info(f"--------- sensitivity: {args.sensitivity} ---------")
train(args.epochs)
sps_tools.print_nonzeros(model, logger=summary_logger)


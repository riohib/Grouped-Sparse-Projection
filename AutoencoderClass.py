from torch import nn
from projection import sparse_opt
import numpy as np
import torch

def sparse(x):
    """
    Preprocesses the Weights to be called by the "sparse_opt" function, from the
    projection.py file.
    :param x: the weight matrix
    :return:
    """
    x_in = x.detach().numpy()
    spar = 0.6
    m = len(x_in[0])
    k = np.sqrt(m) - spar * (np.sqrt(m) - 1)
    x_sparse = np.zeros(np.shape(x_in))

    for i in range(len(x_in)):
        x_sparse[i] = sparse_opt(x_in[i], k)

    cw = torch.tensor(x_sparse, dtype=torch.float32, requires_grad=True)
    # print("Calls the Sparse!")
    return cw

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # encoder
        self.e1 = nn.Linear(28 * 28, 128)
        self.e2 = nn.Linear(128, 64)
        self.e3 = nn.Linear(64, 12)
        self.e4 = nn.Linear(12, 3)
        self.relu = nn.ReLU(True)

        # decoder
        self.d1 = nn.Linear(3, 12)
        self.d2 = nn.Linear(12, 64)
        self.d3 = nn.Linear(64, 128)
        self.d4 = nn.Linear(128, 28*28)

        self.tanh = nn.Tanh()

    def forward(self, x):
        #== (Comment this block out for regular Autoencoder)==
        #======= Sparsity  ==========
        tmp = self.e1.weight
        tmp2 = sparse(tmp)
        self.e1.weight.data = tmp2
        
        #======= Regular Autoencoder Below ==========
        # encode
        x = self.relu(self.e1(x))
        x = self.relu(self.e2(x))
        x = self.relu(self.e3(x))
        x = self.relu(self.e4(x))

        # decode
        x = self.relu(self.d1(x))
        x = self.relu(self.d2(x))
        x = self.relu(self.d3(x))
        x = self.tanh(self.d4(x))
        return x

import torch
import numpy as np
from numpy import linalg as LA
import pickle
import scipy.io
import logging
import pdb


def sparsity(matrix):
    r =  matrix.shape[1] # no of vectors

    spx = 0
    spxList = []
    for i in range(r):
        if matrix[:,i].sum() == 0:
            spx = 1
            spxList.append(spx)
            print('here')
        else:
            ni = matrix.shape[0]
            # spx + ( np.sqrt(ni) - LA.norm(matrix[:, i], 1)/LA.norm(matrix[:, i], 2)) / (np.sqrt(ni)-1)
            spx = ( np.sqrt(ni) - torch.norm(matrix[:, i], 1)/torch.norm(matrix[:, i], 2)) / (np.sqrt(ni)-1)
            spxList.append(spx)            
        spx = sum(spxList)/r

    return spx


def checkCritical(vector, critval_list, precision = 1e-6):

    max_element = max(vector).item()
    num_crit_points = torch.sum(abs(vector - max_element) < precision)
    
    if num_crit_points > 1:
        critval_list.append(max_element)
        print('Crit Point!')
    
    return critval_list, max_element


def gmu(matrix, mu = 0):
 
    vgmu = 0
    gradg = 0
    matrix = torch.abs(matrix)
    xp_vec = torch.zeros([matrix.shape[0], matrix.shape[1]])
    # print('Value of my: '+ str(mu))
    glist = []

    gsp_iter = 0
    for i in range(matrix.shape[1]):
        ni = matrix[:,i].shape[0]
        betai = 1 / (torch.sqrt(torch.tensor(ni, dtype=torch.float64))  - 1)
        #xp_vec = np.concatenate((xp_vec, (matrix[:, i] - mu * betai).reshape(-1, 1)), axis=1)
        xp_vec[:, i] = matrix[:, i] - mu * betai
        indtp = torch.where(xp_vec[:, i] > 0)[0]

        xp_vec[:, i] = torch.relu(xp_vec[:, i])
        # xp_vec[:, i] = torch.max(0, xp_vec[:, i])

        # Save xp_vec:
        gsp_iter += 1
        # print("GSP Iter: " + str(gsp_iter))
        # Outputs
        f2 = torch.norm(xp_vec[:,i])
        if f2 > 0:
            nip = len(indtp)  #may be needs change here
            ev = torch.ones((nip, 1))

            term2 = torch.pow(torch.matmul(ev.T, xp_vec[indtp, i]), 2)
            new_grad = torch.pow(betai, 2) * (-nip * torch.pow(f2, -1) + term2 * torch.pow(f2, -3))
            gradg = gradg + new_grad
            glist.append(new_grad.item())

        if indtp.numel() != 0:
            vec_norm = torch.norm(xp_vec[:, i])
        
            if vec_norm == 0:
                scipy.io.savemat('norm_zero.mat', mdict={'arr': xp_vec})

            xp_vec[:, i] = xp_vec[:, i]/vec_norm
            vgmu = vgmu + betai * torch.sum(xp_vec[:, i])
            # glist.append((betai * torch.sum(xp_vec[:, i])).item() )

        else:
            im = torch.argmax(matrix[:, 0])
            xp_vec[im, i] = 1
            vgmu = vgmu + betai
            # glist.append(betai.item())
        
        # if torch.isnan(xp_vec[:, i]).sum():
        #     pdb.set_trace()

    return vgmu, xp_vec, gradg


def groupedsparseproj(matrix, sps, itr, precision=1e-6, linrat=0.9):
    # sps = 0.9 ;  precision=1e-6; linrat=0.9
    
    epsilon = 10e-15
    k = 0
    muup0 = 0
    r = matrix.shape[1]  # No of Columns

    critmu = torch.tensor([])
    critval_list = []

    # maxxi_list = []

    # These operations were inside the loop, but doesn't need to be.
    matrix_sign = torch.sign(matrix)
    pos_matrix = matrix_sign * matrix
    ni = matrix.shape[0]

    for i in range(r):
        k = k + np.sqrt(ni)/(np.sqrt(ni) - 1)
        # check critical values of mu where g(mu) is discontinuous, that is,
        # where the two (or more) largest entries of x{i} are equal to one another.

        critical_val, max_xi = checkCritical(pos_matrix[:,i], critval_list)
        
        # maxxi_list.append(max_xi)

        muup0 = max(muup0, max_xi * (np.sqrt(ni)-1))
        critmu = torch.tensor(critval_list) * (np.sqrt(ni) - 1)

    k = k - r * sps
    vgmu, xp_vec, gradg  = gmu(pos_matrix, 0)


    if vgmu < k:
        xp_mat = matrix
        gxpmu = vgmu
        numiter = 0
        return xp_mat
    else:
        numiter = 0
        mulow = 0
        glow = vgmu
        muup = muup0
        # Initialization on mu using 0, it seems to work best because the
        # slope at zero is rather steep while it is gets falt for large mu
        newmu = 0
        gnew = glow
        gpnew = gradg         # g'(0)
        delta = muup - mulow
        switch  = True 

        while abs(gnew - k) > precision * r and numiter < 100:
            oldmu = newmu
            # % Secant method: 
            # % newmu = mulow + (k-glow)*(muup-mulow)/(gup-glow);

            # % Bisection: 
            # % newmu = (muup+mulow)/2;
            # % Newton: 
            newmu = oldmu + (k - gnew) / (gpnew + epsilon) 

            if (newmu >= muup) or (newmu <= mulow): #If Newton goes out of the interval, use bisection
                newmu = (mulow+muup)/2
            
            # print( 'Value of numiter: ' + str(numiter))
            gnew, xnew, gpnew = gmu(matrix, newmu)

            if gnew < k:
                gup = gnew
                xup = xnew
                muup = newmu
            else:
                glow = gnew
                mulow = xnew
                mulow = newmu

            # Guarantees linear convergence
            if (muup - mulow) > linrat * delta and abs(oldmu-newmu) < (1-linrat)* delta:
                newmu = (mulow + muup) / 2
                gnew, xnew, gpnew = gmu(matrix, newmu)

                if gnew < k:
                    gup = gnew
                    xup = xnew
                    muup = newmu
                else:
                    glow = gnew
                    mulow = xnew
                    mulow = newmu
                numiter += 1
            numiter += 1

            if critmu.shape[0] != 0 and abs(mulow-muup) < abs(newmu)*precision and \
                                min( abs(newmu - critmu) ) < precision*newmu:
                print('The objective function is discontinuous around mu^*.')
                xp = xnew
                gxpmu = gnew
        try:
            xp_vec = xnew
            # print(' xp_vec = xnew')
        except:
            scipy.io.savemat('matrix.mat', mdict={'arr': matrix})
            
        gxpmu = gnew

    # pdb.set_trace()

    alpha = torch.zeros([1, matrix.shape[1]])
    for i in range(r):
        alpha[0, i] = torch.matmul(xp_vec[:, i], pos_matrix[:, i])
        xp_vec[:, i] = alpha[:, i] * (matrix_sign[:, i] * xp_vec[:, i])

    return xp_vec

def load_matrix_debug():
    with open("test_matrix.txt", "rb") as fpA:   #Pickling
        matrix  = pickle.load(fpA)
    return matrix

## ********************************************************************************** ##

# r = 100
# n = 10000
# k = 0

# ## Data Loacing
# # mu, sigma = 0, 1 # mean and standard deviation
# # x = np.random.normal(mu, sigma, (10000, 100)) * 10

# with open('matnew.pkl', 'rb') as fin:
#     x = pickle.load(fin)
# ## ****************************************

# xPos = np.abs(x)

# for i in range(r):
#     k = k + np.sqrt(n) / (np.sqrt(n) - 1)

# sp = sparsity(x)

# print("The Sparsity of input set of vectors: " + str(sp))

# xp_vec = groupedsparseproj(x, 0.8)

# spNew = sparsity(xp_vec)

# print("The Output Sparsity: " + str(spNew))


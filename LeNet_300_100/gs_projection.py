import torch
import numpy as np
from numpy import linalg as LA
import pickle
import scipy.io
import logging


def sparsity(matrix):
    r =  matrix.shape[1] # no of vectors

    spx = 0
    for i in range(r):
        if matrix[:,1].sum() == 0:
            spx = 1
        else:
            ni = matrix.shape[0]
            spx = spx + ( np.sqrt(ni) - LA.norm(matrix[:, i], 1)/LA.norm(matrix[:, i], 2)) / (np.sqrt(ni)-1)
            # spx = spx + ( np.sqrt(ni) - torch.norm(matrix[:, i], 1)/torch.norm(matrix[:, i], 2)) / (np.sqrt(ni)-1)
    spx = spx/r
    return spx


def checkCritical(vector, precision = 1e-6):
    crit_values = []
    max_element = max(vector)
    index = np.where(abs(vector - max_element) < precision)
    if len(index[0]) > 1:
        crit_values.append(max_element)
        # crit_values = max_element
    else:
        crit_values = []
    return (crit_values, max_element)


def gmu(matrix, mu = 0):
    vgmu = 0
    gradg = 0
    matrix = np.abs(matrix)
    xp_vec = np.empty([matrix.shape[0], matrix.shape[1]])

    gsp_iter = 0
    for i in range(matrix.shape[1]):
        ni = matrix[:,i].shape[0]
        betai = 1 / (np.sqrt(ni) - 1)
        #xp_vec = np.concatenate((xp_vec, (matrix[:, i] - mu * betai).reshape(-1, 1)), axis=1)
        xp_vec[:, i] = matrix[:, i] - mu * betai
        indtp = np.where(xp_vec[:, i] > 0)[0]
        xp_vec[:, i] = np.maximum(0, xp_vec[:, i])

        # Save xp_vec:
        xp_vec[:, i] = np.maximum(0, xp_vec[:, i])
        gsp_iter += 1
        # print("GSP Iter: " + str(gsp_iter))
        # Outputs
        f2 = LA.norm(xp_vec)
        if f2 > 0:
            nip = len(indtp)
            ev = np.ones((nip, 1))

            term2 = np.power(np.sum(ev.T * xp_vec[indtp, i], axis=1), 2)
            gradg = gradg + np.power(betai, 2) * (-nip * np.power(f2, -1) + term2 * np.power(f2, -3)) 

        if indtp.size != 0:
            normVect = LA.norm(xp_vec[:, i])
            if normVect == 0:
                scipy.io.savemat('norm_zero.mat', mdict={'arr': xp_vec})

            xp_vec[:, i] = xp_vec[:, i]/normVect
            vgmu = vgmu + betai * sum(xp_vec[:, i])
        else:
            im = np.argmax(matrix[:, 0])
            xp_vec[im, i] = 1
            vgmu = vgmu + betai

    return vgmu, xp_vec, gradg


def groupedsparseproj(matrix, sparsity, itr, precision=1e-6, linrat=0.9):
    epsilon = 10e-15
    k = 0
    muup0 = 0
    r = matrix.shape[1]  # No of Columns
    critmu = np.array([])

    # These operations were inside the loop, but doesn't need to be.
    matrix_sign = np.sign(matrix)
    pos_matrix = matrix_sign * matrix
    ni = matrix.shape[0]

    for i in range(r):
        # matrix_sign = np.sign(matrix)
        # pos_matrix = matrix_sign * matrix
        # ni = matrix.shape[0]
        k = k + np.sqrt(ni)/(np.sqrt(ni) - 1)
        # check critical values of mu where g(mu) is discontinuous, that is,
        # where the two (or more) largest entries of x{i} are equal to one another.

        return_tuple = checkCritical(matrix[:,i])
        critical_val, max_xi = return_tuple
        muup0 = max(muup0, max_xi * (np.sqrt(ni)-1))
        #critmu.append(list(np.array(critical_val) * (np.sqrt(ni) - 1)))
        critmu = np.concatenate((critmu, np.array(critical_val) * (np.sqrt(ni) - 1)))

    k = k - r * sparsity
    vgmu, xp_vec, gradg  = gmu(matrix, 0)


    if vgmu < k:
        xp_mat = matrix
        gxpmu = vgmu
        muup = muup0
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
            newmu = oldmu + (k - gnew) / (gpnew + epsilon) 
            
            # if (itr % 25 == 0) and switch:
            #     logging.debug("newmu: %.4f | oldmu: %.4f | k: %.4f | gnew: %.4f |  gpnew: %.4f |\n" % 
            #             (newmu, oldmu, k, gnew, gpnew))
            #     switch = False

            if (newmu >= muup) or (newmu <= mulow): #If Newton goes out of the interval, use bisection
                newmu = (mulow+muup)/2
            gnew, xnew, gpnew = gmu(matrix, newmu)

            if gnew < k:
                gup = gnew
                xup = xnew
                muup = newmu
            else:
                glow = gnew
                mulow = xnew
                mulow = newmu

            # Garantees linear convergence
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
        except:
            scipy.io.savemat('matrix.mat', mdict={'arr': matrix})
            
        gxpmu = gnew

    alpha = np.zeros([1, matrix.shape[1]])
    for i in range(r):
        alpha[0, i] = xp_vec[:, i].T @ pos_matrix[:, i]
        xp_vec[:, i] = alpha[:, i] * (matrix_sign[:, i] * xp_vec[:, i])

    return xp_vec

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


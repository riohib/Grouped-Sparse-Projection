# The Sparsity Function is being called inside the Training Loop


from AutoencoderSimple import Autoencoder
from projection import sparse_opt
from gs_projection import *

import logging
#************************* Imports *************************

import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from matplotlib import pyplot as plt
import numpy as np
import pickle

np.random.seed(0)
torch.manual_seed(0)
#************************* Imports *************************

logging.basicConfig(filename='LogFile.log', level=logging.DEBUG)


list = []
nan_list = []
nan_iter = []

def weight_splot(version):
    fig = plt.figure(4)
    weightr = model.e1.weight.detach()
    c = 0
    for i in range(128):
        plt.subplot(12, 12, c + 1)
        plt.imshow(weightr[c].view(28, 28), cmap=plt.cm.RdBu_r)
        plt.clim(-1, 1)
        c += 1
        plt.axis('off')
    plt.savefig('orderedWeights{}.png'.format(version))


def gsp(model, iter):
    sps = 0.8

    w1 = model.e1.weight.detach().numpy()

    # if (iter == 48):
    #     scipy.io.savemat('w1.mat', mdict={'arr': w1})

    sparse_w1 = groupedsparseproj(w1, sps, iter)

    model.e1.weight.data = torch.tensor(sparse_w1, dtype=torch.float32)
    
    if iter % 10 == 0:
        logging.debug(" ------------------- Iter No: %s ------------------ \n" % iter)
        logging.debug("Layer 1 Sparsity w1 | before: %.2f | After: %.2f \n" % 
                                (sparsity(w1), sparsity(model.e1.weight.detach().numpy())))

    # print("---------------------------------------------------")


def sparsity_prev(model):
    """
    Preprocesses the Weights to be called by the "sparse_opt" function, from the
    projection.py file.
    :param x: the weight matrix
    :return:
    """
    spar = 0.7

    w1 = model.e1.weight.detach()
    w2 = model.e2.weight.detach()
    w3 = model.e3.weight.detach()

    x_L1 = torch.nn.functional.relu(w1, inplace=False).numpy()
    x_L2 = torch.nn.functional.relu(w2, inplace=False).numpy()
    x_L3 = torch.nn.functional.relu(w3, inplace=False).numpy()

    m1 = np.prod(x_L1.shape)
    m2 = np.prod(x_L2.shape)
    m3 = np.prod(x_L3.shape)
    mtot = m1 + m2 + m3

    x_sparse = np.zeros(mtot)
    x_sparse1 = np.zeros(m1)
    x_sparse2 = np.zeros(m2)
    x_sparse3 = np.zeros(m3)

    k = np.sqrt(mtot) - spar * (np.sqrt(mtot) - 1)

    a = x_L1.flatten().reshape(1, -1)
    b = x_L2.flatten().reshape(1, -1)
    c = x_L3.flatten().reshape(1, -1)
    all_w = np.concatenate((a, b, c), axis=None)

    #a = sparse_opt(np.sort(-x_L1.flatten())[::-1], k)

    a = sparse_opt(np.sort(-all_w)[::-1], k)

    #ind = np.argsort(-x_L1.flatten())[::-1]
    ind = np.argsort(-all_w)[::-1]
    x_sparse[ind] = a

    x_sparse1 = x_sparse[0:m1]
    x_sparse2 = x_sparse[m1:m1+m2]
    x_sparse3 = x_sparse[m1+m2:mtot]

    x_sparse1 = np.reshape(x_sparse1, x_L1.shape)
    x_sparse2 = np.reshape(x_sparse2, x_L2.shape)
    x_sparse3 = np.reshape(x_sparse3, x_L3.shape)

    cw1 = torch.tensor(x_sparse1, dtype=torch.float32, requires_grad=True)
    cw2 = torch.tensor(x_sparse2, dtype=torch.float32, requires_grad=True)
    cw3 = torch.tensor(x_sparse2, dtype=torch.float32, requires_grad=True)

    model.e1.weight.data = cw1
    model.e2.weight.data = cw2
    model.e3.weight.data = cw3
    #return cw

def force_weight_positive(model):
    """
    Simply Forcing the weights to stay positive without sparsity
    to check whether clamped positive weights make the Autoencoder learn.
    It does!
    """
    w = model.e1.weight.detach()
    w_pos = torch.nn.functional.relu(w, inplace=False).numpy()  # Make Weights positive only
    cw = torch.tensor(w_pos, dtype=torch.float32, requires_grad=True)
    model.e1.weight.data = cw

#******************** Result Directories **************************
if not os.path.exists('./layer_image'):
    os.mkdir('./layer_image')
if not os.path.exists('./Loss'):
    os.mkdir('./Loss')
if not os.path.exists('./weights_plot'):
    os.mkdir('./weights_plot')
if not os.path.exists('./compare'):
    os.mkdir('./compare')


def weight_plot(iter):
    plt.figure(2)
    weights = model.e1.weight.detach()
    io = to_img(weights)
    plt.imshow(io[2].view(28, 28))
    save_image(io, './weights_plot/wp_{}.png'.format(iter))


#**************************** Imports ******************************
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

#************************* Data Processing *************************
num_epochs = 15
batch_size = 64
learning_rate = 1e-3

img_transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])

dataset = MNIST('./data', download=True, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#************************* Model Setup *************************
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


#************************* Training Loop *************************
loss_array = []
l1_weights = []
iter = 0
in_sparse = 0
for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_array.append(loss)

        # ===================Sparsity Enforcement====================
        if iter % 1 == 0:
            in_sparse += 1
            gsp(model, iter)
            #force_weight_positive(model)

        if iter % 300 == 0:
            pic = to_img(output.data)
            save_image(data[0], './compare/original_{}.png'.format(iter))
            save_image(pic, './compare/reconstructed_{}.png'.format(iter))


        iter += 1
        print(iter)
        print("---------------------------------------------------")
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './layer_image/model_output{}.png'.format(epoch))
    #     # #========================= Log Weights ========================
    #     # weights = model.e1.weight.detach().numpy()
    #     # l1_weights.append(weights)
    #     # np.savetxt("./Loss/l1_il_weights_ep{}.csv".format(epoch), weights, delimiter=",")

torch.save(model.state_dict(), './gsp_ae.pth')

fig=plt.figure(0)
plt.plot(loss_array)
with open("./Loss/loss.txt", "wb") as fp:   #Pickling
    pickle.dump(loss_array, fp)

fig.savefig('./Loss/lossCurve.png', bbox_inches='tight', dpi=100)

# ========================== Log loss_array to file ===================================
def save_loss(loss_array):
    la=[]
    for l in loss_array:
        tmp = l.detach().numpy()
        la.append(tmp)
        np.savetxt("./Loss/loss_s.csv", la, delimiter=",")

save_loss(loss_array)

# ************************** Original & Reconstructed Image ****************************
def next_image():
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for data in dataloader:
        data1 = data
        break
    return data1

def compare(data1, model, im_num):
    """
    Shows Original vs Re-constructed Image
    """
    im_num = 26
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(data1[0][im_num].view(28,28))

    # Plot Reduced Representation
    img, _ = data1
    img = img.view(img.size(0), -1)
    img = Variable(img)
    out = model(img)

    plt.figure(1)
    plt.subplot(1,2,2)
    result = out[im_num].detach()
    plt.title('Reconstructed')
    plt.imshow(result.view(28,28))
    plt.show()

# data1 = next_image()
# compare(data1, 11)


# ===== Plot L1 ===============
plt.figure(2)
weights = model.e1.weight.detach()
io = to_img(weights)
plt.imshow(io[20].view(28,28))
save_image(io, './weights_plot/wp.png'.format(epoch))


print( "Any negative number in L1 weights: " + str(True in list))
print("Total Iterations: " + str(iter) )
print("Sparsity Applied: " + str(in_sparse) )
print("Any Nan: " + str(True in nan_list))
print("nan in iter: " + str(nan_iter))


# weight_splot(1)


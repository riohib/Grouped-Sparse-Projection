# The Sparsity Function is being called inside the Training Loop


from AutoencoderSimple import Autoencoder
from projection import sparse_opt
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

list = []
nan_list = []
nan_iter = []

#******************** Sparsity *****************************
# def sparsity(model, iter):
#
#     ww = model.e1.weight.detach()
#     x_in = torch.nn.functional.relu(ww, inplace=False).numpy()  # Make Weights positive only
#     #================ for testing of Negative weights/ nan ==================
#     list.append(np.any(x_in < 0))
#     if np.isnan(x_in).any():
#         nan_list.append(np.isnan(x_in).any())
#         nan_iter.append(iter)
#
#     # x_in = model.e1.weight.detach().numpy()
#     spar = 0.6
#     m = len(x_in[0])
#     k = np.sqrt(m) - spar * (np.sqrt(m) - 1)
#     x_sparse = np.zeros(np.shape(x_in))
#
#     for i in range(len(x_in)):
#         x_sparse[i] = sparse_opt(x_in[i], k)
#
#     cw = torch.tensor(x_sparse, dtype=torch.float32, requires_grad=True)
#     model.e1.weight.data = cw


def sparsity(model):
    """
    Preprocesses the Weights to be called by the "sparse_opt" function, from the
    projection.py file.
    :param x: the weight matrix
    :return:
    """

    ww = model.e1.weight.detach()
    x_in = torch.nn.functional.relu(ww, inplace=False).numpy()
    spar = 0.99

    m = np.prod(x_in.shape)

    x_sparse = np.zeros(m)
    k = np.sqrt(m) - spar * (np.sqrt(m) - 1)

    a = sparse_opt(np.sort(-x_in.flatten())[::-1], k)
    ind = np.argsort(-x_in.flatten())[::-1]

    x_sparse[ind] = a

    x_sparse = np.reshape(x_sparse, x_in.shape)

    cw = torch.tensor(x_sparse, dtype=torch.float32, requires_grad=True)
    #     # print("Calls the Sparse!")
    return cw

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
if not os.path.exists('./mlp_img_inloop'):
    os.mkdir('./mlp_img_inloop')
if not os.path.exists('./Loss_loop'):
    os.mkdir('./Loss_loop')
if not os.path.exists('./weights_plot'):
    os.mkdir('./weights_plot')

def weight_plot(iter):
    plt.figure(2)
    weights = model.e1.weight.detach()
    io = to_img(weights)
    plt.imshow(io[2].view(28, 28))
    save_image(io, './weights_plot/wp_{}.png'.format(iter))

#************************* Imports *************************

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

#************************* Data Processing *************************
num_epochs = 5
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
iter = 1
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
            sparsity(model)
            # force_weight_positive(model)

        if iter % 100 == 0:
            weight_plot(iter)
            pic = to_img(output.cpu().data)
            save_image(pic, './mlp_img_inloop/image_l_{}.png'.format(epoch))

        iter += 1
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))
    if epoch % 10 == 0:
        # pic = to_img(output.cpu().data)
        # save_image(pic, './mlp_img_inloop/image_l_{}.png'.format(epoch))
        #========================= Log Weights ========================
        weights = model.e1.weight.detach().numpy()
        l1_weights.append(weights)
        np.savetxt("./Loss_loop/l1_il_weights_ep{}.csv".format(epoch), weights, delimiter=",")

torch.save(model.state_dict(), './sim_autoencoder.pth')

fig=plt.figure(0)
plt.plot(loss_array)
fig.savefig('./Loss_loop/loss_sp_curve.png', bbox_inches='tight', dpi=100)

# ========================== Log loss_array to file ===================================
def save_loss(loss_array):
    la=[]
    for l in loss_array:
        tmp = l.detach().numpy()
        la.append(tmp)
        np.savetxt("./Loss_loop/loss_s.csv", la, delimiter=",")

save_loss(loss_array)

# ************************** Original & Reconstructed Image ****************************
def next_image():
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for data in dataloader:
        data1 = data
        break
    return data1

def compare(data1):
    """
    Shows Original vs Re-constructed Image
    """
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(data1[0][5].view(28,28))

    # Plot Reduced Representation
    img, _ = data1
    img = img.view(img.size(0), -1)
    img = Variable(img)
    out = model(img)

    plt.figure(1)
    plt.subplot(1,2,2)
    result = out[0].detach()
    plt.title('Reconstructed')
    plt.imshow(result.view(28,28))
    plt.show()

data1 = next_image()
compare(data1)


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

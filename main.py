# The Sparsity Function is being called inside the Training Loop

from AutoencoderClass import Autoencoder
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

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')

torch.manual_seed(10)
#************************* Imports *************************

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

#************************* Data Processing *************************
num_epochs = 10
batch_size = 64
learning_rate = 1e-3

img_transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])

dataset = MNIST('./data', download=True, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#************************* Image Display *************************
# data_iter = iter(dataloader)
# data = next(data_iter)
#data[0][0].shape
#plt.imshow(data[0][0].view(28,28))
#plt.show()

#************************* Model Setup *************************
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


#************************* Training Loop *************************
loss_array = []
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
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './mlp_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')


# ************************** Original & Reconstructed Image ****************************
def next_image():
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for data in dataloader:
        data1 = data
        break
    return data1

def compare(data1):
    """
    Shows Original vs Re constructed Image
    """
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(data1[0][0].view(28,28))

    # Plot Reduced Representation
    img, _ = data1
    img = img.view(img.size(0), -1)
    img = Variable(img)

    out = model(img)
    plt.subplot(1,2,2)
    result = out[0].detach()
    plt.title('Reconstructed')
    plt.imshow(result.view(28,28))
    plt.show()

data1 = next_image()
compare(data1)
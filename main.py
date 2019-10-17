# The Sparsity Function is being called inside the Training Loop


from AutoencoderClass import Autoencoder
from projection import *
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

#************************* Imports *************************

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

#************************* Data Processing *************************
num_epochs = 1
batch_size = 64
learning_rate = 1e-3

img_transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])

dataset = MNIST('./data', download=True, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#************************* Image Display *************************
data_iter = iter(dataloader)
data = next(data_iter)
data[0][0].shape
plt.imshow(data[0][0].view(28,28))
plt.show()

#************************* Model Setup *************************
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


#************************* Training Loop *************************
loss_array = []
iter = 0
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
        # ===================Sparsity Enforcement====================
        x_in = model.e1.weight.detach().numpy()
        x_sparse = LinearHoyer(x_in, 784, 150, 0.9)
        cw = torch.tensor(x_sparse, dtype=torch.float32, requires_grad=True)
        model.e1.weight.data = cw

        loss_array.append(loss)

        # ================== Break after 3 loops =====================
        print(iter)
        iter += 1
        if iter == 3:
            break
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './mlp_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')


plt.plot(loss_array)
plt.show()


# ************************** Original & Reconstructed Image ****************************
data_iter = iter(dataloader)
data = next(data_iter)


# Plot Original
plt.subplot(1,2,1)
plt.title('Original')
plt.imshow(data[0][0].view(28,28))

# Plot Reduced Representation
img, _ = data
img = img.view(img.size(0), -1)
img = Variable(img)

out = model(img)
plt.subplot(1,2,2)
result = out[0].detach()
plt.title('Reconstructed')
plt.imshow(result.view(28,28))
plt.show()
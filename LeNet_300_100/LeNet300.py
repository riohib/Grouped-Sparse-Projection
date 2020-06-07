import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from matplotlib import pyplot as plt
import logging

from helper import *

## New Post Conf

#******************** Result Directories **************************
if not os.path.exists('./Loss'):
    os.mkdir('./Loss')
if not os.path.exists('./weights_plot'):
    os.mkdir('./weights_plot')
    
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========================== GSP FUNCTION ==================================

def gsp(model, iter):
    sps = 0.85

    w1 = model.e1.weight.detach().numpy()
    w2 = model.e2.weight.detach().numpy()

    # if (iter == 48):
    #     scipy.io.savemat('w1.mat', mdict={'arr': w1})

    sparse_w1 = groupedsparseproj(w1, sps, iter)
    sparse_w2 = groupedsparseproj(w2, sps, iter)

    model.e1.weight.data = torch.tensor(sparse_w1, dtype=torch.float32)
    model.e2.weight.data = torch.tensor(sparse_w2, dtype=torch.float32)
    
    if iter % 10 == 0:
        logging.debug(" ------------------- Iter No: %s ------------------ \n" % iter)
        logging.debug("Layer 1 Sparsity w1 | before: %.2f | After: %.2f \n" % 
                        (sparsity(w1), sparsity(model.e1.weight.detach().numpy())))
        logging.debug("Layer 2 Sparsity w2 | before: %.2f | After: %.2f \n" % 
                        (sparsity(w2), sparsity(model.e2.weight.detach().numpy())))

# ========================== GSP FUNCTION ==================================

# Hyper-parameters 
input_size = 784
hidden_size1 = 300
hidden_size2 = 100
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()   # Runs the __init__ of the parent class.

        self.e1 = nn.Linear(28*28, 300)  # flattened image dim is the input = 28*28, output=64
        self.e2 = nn.Linear(300, 100)
        self.e3 = nn.Linear(100, 10)
    
    def forward(self, x):
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        x = self.e3(x)
        return F.softmax(x, dim=1)


# model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes).to(device)
model = Net().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
loss_array = []
in_sparse = 0
iter = 0
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_array.append(loss)

        if iter % 15 == 0:
            in_sparse += 1
            gsp(model, iter)

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        iter += 1

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
torch.save(model.state_dict(), './LeNet300.pth')


#============================ POST PROCESSING ================================

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


# ================================ Plot L1 ======================================
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


sModel = Net()
sModel.load_state_dict(torch.load('LeNet300.pth', map_location='cpu'))
weight_splot(1, sModel)
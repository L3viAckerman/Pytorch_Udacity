import torch 
import numpy as np 
from torchvision import datasets
import torchvision.transforms as transforms

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load data training and test datasets
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

num_workers = 0
batch_size = 20

train_on_gpu = torch.cuda.is_available()

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

## Visualize the Data

import matplotlib.pyplot as plt

datatier = iter(train_loader)
images, lables = datatier.next()
images = images.numpy()
print(images.shape)

# get one image from the batch
img = np.squeeze(images[2])

fig  = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
# plt.show()

## Convolution autoencoder
import torch.nn as nn
import torch.nn.functional as F

# defind the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # depth from 1 --> 16
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1) # depth from 16 --> 4
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        ## encode ##
        x = F.relu(self.dropout(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)

        ## decode ##
        x = F.relu(self.t_conv1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.t_conv2(x)) # output layer with sigmoid for scaling from 0 -> 1

        return x

# initialize the NN
model = ConvAutoencoder()
print(model)
if train_on_gpu:
    model = model.cuda()



## Training

# specify loss function
criterion = nn.MSELoss()

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

n_epochs = 100

for epoch in range(1, n_epochs+1):
    train_loss = 0.0

    for data in train_loader:
        ## train model ##
        images, _ = data
        if train_on_gpu:
            images = images.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass
        outputs = model(images)
        # calculate loss
        loss = criterion(outputs, images)
        # backward pass
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining loss: {:.6f}'.format(epoch, train_loss))


## Checking out the results

datatier = iter(test_loader)
images, lables = datatier.next()

# get sample outpust
images = images.cuda()
output = model(images)
# prepare images for display

output = output.view(batch_size, 1, 28, 28)
# use detach when it's an output that requires_grad
output = output.cpu()
output = output.detach().numpy()
images = images.cpu()

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()

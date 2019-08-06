## Generative Adverarial Network
# Buidling a Generative adversarial network (GAN) trained on the MNIST datasets. From this, we will be able to generative new handwritten digits!
# The idea behind GAN is that you have two networks, a generator G an a discriminator D, competing against each other.
# The generator makes 'fake' data to pass to the discriminator. The discriminator also sees real training data and predicts if the data it's reveived is real or fake.


import numpy as np 
import torch
import matplotlib.pyplot as plt 

from torchvision import datasets, transforms

num_workers = 0
batch_size = 64

# covert data to torch.FloatTensor
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', download=True, train=True, transform=transform)

# prepare data loader 
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

# Visualize the data
datatier = iter(train_loader)
images, labels = datatier.next()
images = images.numpy()

img = np.squeeze(images[0])

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
# plt.show()

## Define the model

import torch.nn as nn
import torch.nn.functional as F 

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()

        # define hidden linear layers
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)

        # final fully-connected layer
        self.fc4 = nn.Linear(hidden_dim, output_size)
        
        self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        # flatten image
        x = x.view(-1, 28 * 28)
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slop=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # final layer
        out = self.fc4(x)

        return out

class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)

        # final fully-connected layer 
        self.fc4 = nn.Linear(hidden_dim * 4, output_size)

        # dropout layer 
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)

        out = F.tanh(self.fc4(x))

        return out

# Model hyperparameters 

# Discriminator hyperparameters
input_size = 784
d_output_size = 1 # size of discriminator output (real or fake)
d_hidden_size = 32

# Generator hyperparameters
z_size = 100
# size of discriminator output 
g_output_size = 784
g_hidden_size = 32

## Build complete network 

D = Discriminator(input_size, d_hidden_size, d_output_size)
G = Generator(z_size, g_hidden_size, g_output_size)

print(D)
print()
print(G)


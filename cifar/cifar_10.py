import torch 
import numpy as np 

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available. Training on GPU ...')

## Load the Data

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

num_workers = 0

batch_size = 20

valid_size = 0.2

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

# obtain traning indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(num_train * valid_size))
train_idx, valid_idx = indices[: split], indices[split: ]

# define sampler for obtainng training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders for obtaining training and validation batches
# lables is the array with value is from 0 to 9
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, sampler=valid_sampler, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

## Visualize a batch of training data

import matplotlib.pyplot as plt 

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))    # convert from Tensor image

# obtain one batch of traning images
datatier = iter(train_loader)
images, lables = datatier.next()
images = images.numpy()     # convert images to numpy for display

# plot the image in the batch, along with the corresponding lables
fig = plt.figure(figsize=(25, 4))
# display 20 image
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx + 1, xticks = [], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[lables[idx]])

# plt.show()

## Define Network Architecture
import torch.nn as nn
import torch.nn.functional as F

class CNN_Cifar10(nn.Module):
    def __init__(self):
        super(CNN_Cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * 4 *4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

# create a complete CNN model
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * 4 *4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

# create a complete CNN model
model = CNN_Cifar10()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model = model.cuda()

# specify Loss Func and Optimizer
import torch.optim as optim
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer 
optimizer = optim.SGD(model.parameters(), lr = 0.01)

## Train the network

# n_epochs = 30

# valid_loss_min = np.Inf

# for epoch in range(1, n_epochs + 1):

#     # keep track of training and validation loss
#     train_loss = 0
#     valid_loss = 0

#     # train
#     model.train()
#     for data, target in train_loader:
#         if train_on_gpu:
#             data, target = data.cuda(), target.cuda()
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()*data.size(0)

#     model.eval()
#     for data, target in valid_loader:
#         if train_on_gpu:
#             data, target = data.cuda(), target.cuda()
#         output = model(data)
#         loss = criterion(output, target)
#         valid_loss += loss.item()*data.size(0)
    

#     # calculate averange losses
#     train_loss = train_loss / len(train_loader.sampler)
#     valid_loss = valid_loss / len(valid_loader.sampler)

#     print('Epcoch : {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
    
#     # save model if validation loss has decreased
#     if valid_loss < valid_loss_min:
#         torch.save(model.state_dict(), 'model_cifar10.pt')
#         valid_loss_min = valid_loss


# load the model with the lowest validation loss
model.load_state_dict(torch.load('model_cifar10.pt'))

# Test the Train network

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for data, target in test_loader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
        print(target)
        # output shape is 20*10 with 20 is the batch size and 10 is the number of classes
        output = model(data)
        print(output)
        loss = criterion(output, target)
        test_loss += loss.item()*data.size(0)
        # convert output propabilities to predicted class
        print(torch.max(output, dim = 1))
        break

model = CNN_Cifar10()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model = model.cuda()

# specify Loss Func and Optimizer
import torch.optim as optim
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer 
optimizer = optim.SGD(model.parameters(), lr = 0.01)

## Train the network

# n_epochs = 30

# valid_loss_min = np.Inf

# for epoch in range(1, n_epochs + 1):

#     # keep track of training and validation loss
#     train_loss = 0
#     valid_loss = 0

#     # train
#     model.train()
#     for data, target in train_loader:
#         if train_on_gpu:
#             data, target = data.cuda(), target.cuda()
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()*data.size(0)

#     model.eval()
#     for data, target in valid_loader:
#         if train_on_gpu:
#             data, target = data.cuda(), target.cuda()
#         output = model(data)
#         loss = criterion(output, target)
#         valid_loss += loss.item()*data.size(0)
    

#     # calculate averange losses
#     train_loss = train_loss / len(train_loader.sampler)
#     valid_loss = valid_loss / len(valid_loader.sampler)

#     print('Epcoch : {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
    
#     # save model if validation loss has decreased
#     if valid_loss < valid_loss_min:
#         torch.save(model.state_dict(), 'model_cifar10.pt')
#         valid_loss_min = valid_loss


# load the model with the lowest validation loss
model.load_state_dict(torch.load('model_cifar10.pt'))

# Test the Train network

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for data, target in test_loader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
        print(target)
        # output shape is 20*10 with 20 is the batch size and 10 is the number of classes
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()*data.size(0)
        # convert output propabilities to predicted class
        _, pred = torch.max(output, dim = 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    

# averange test loss
test_loss = test_lost / len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (classes[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
        
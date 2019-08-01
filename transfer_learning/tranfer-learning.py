## Transfer Learning
# Most of the time you won't want to train a whole convolution neural network yourself. Modern ConvNets training on hug datasets like ImageNet take week
# on multiple GPUs. In here, I will be using VGGNet trained on the ImageNet database as a feature extractor.


## Download data
import os 
import numpy as np 
import torch

import torchvision 
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt 

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Train on CPU')
else:
    print('CUDA is available! Train on GPU')

## Load and transform our data
# using PyTorch ImageFolder class which makes it very easy to load data from a directory.
# For example, the training images are all stored in a directory path that look like: root/class_1/xxx.png


# define training and test data directories
data_dir = 'flower_photo/'
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')

classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Transforming the data

# load and transform data using ImageFolder

# VGG16 Takes 224*224 image as input, so we resize all of them
data_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
train_data = datasets.ImageFolder(train_dir, transform=data_transform)
test_data = datasets.ImageFolder(test_dir, transform=data_transform)

# DataLoaders and Data visulization
batch_size = 30
num_workers = 0

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# Visualize some sample data

datatier = iter(train_loader)
images, lables = datatier.next()

# plot the images in the batch, along with the corresponding labels 
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[lables[idx]])

plt.show()

## Define the model

# Load the pretrained model from pytorch
vgg16 = models.vgg16(pretrained=True)

# print out the model structure
print(vgg16)

print(vgg16.classifier[6].in_features)
print(vgg16.classifier[6].out_features)

# Freeze training for all "features" layers
for param in vgg16.features.parameters():
    param.requires_grad = False


# Final Classifier layer
import torch.nn as nn

n_inputs = vgg16.classifier[6].in_features

# add last linear layer (n_inputs -> 5 flower classes)
# new layers automatically have requires_grad = True
last_layer = nn.Linear(n_inputs, len(classes))

vgg16.classifier[6] = last_layer

# if GPU is available, move the model to GPU
if train_on_gpu:
    vgg16 = vgg16.cuda()

# check to see that your last layer produces the expected number of outputs
print(vgg16.classifier[6].out_features)
# print(vgg16)

# Specify Loss Function and Optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(vgg16.classifier.parameters(), lr = 0.001)

## Training

n_epochs = 2

for epoch in range(1, n_epochs+1):
    train_loss = 0.0

    # train model #
    for batch_i, (data, target) in enumerate(train_loader):
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            outputs = vgg16(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_i % 20 == 19:
                print('Epoch %d, Batch %d loss: %.16f' % (epoch, batch_i+1, train_loss / 20))
                train_loss = 0.0


## Testing
# track test loss
# over 5 flower classes
test_loss = 0.0
class_correct = list(0.0 for i in range(5))
class_total = list(0.0 for i in range(5))

vgg16.eval() # eval mode

# iterate over the test data
for data, target in test_loader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
        outputs = vgg16(data)
        loss = criterion(outputs, target)
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(outputs, dim = 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate  test accuracy for each object class
        for i in range(batch_size):
            lable = target.data[i]
            class_correct[lable] += correct[i].item()
            class_total[lable] += 1
        
# calculate avg test loss 
test_loss = test_loss / len(test_loader.dataset)
print('Test loss: {:.6f}\n'.format(test_loss))

for i in range(5):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% {%2d/%2d)' % (classes[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))


## Visualize sample test results

datatier = iter(test_loader)
images, lables = datatier.next()
images.numpy()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

output = vgg16(images)
# convert output probabilities to predicted class
_, pred_tensor = torch.max(output, dim = 1)
preds = np.squeeze(pred_tensor.numpy()) if not train_on_gpu else np.squeeze(pred_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true lables
fig = flt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx + 1, xticks = [], yticks = [])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[lables[idx]]), color = ('green' if preds[idx] == labels[idx].item() else 'red'))
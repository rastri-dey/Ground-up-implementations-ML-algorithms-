'''
ML Algorithm: Deep Convolutional Neural Network 
Deep Learning Framework: PyTorch
Dataset: CIFAR10

Steps:
# Import Libraries
# Check the device enabled for torch - GPU (or CPU)
# Get the dataset:
    Training Dataset
    Validation Dataset
# Normalize the Dataset:
    Calculate the mean, std of the Datset
    Convert the Dataset into image Tensors
    Normalize the image tensors with mean & std
    Do this for both Training and Validation Dataset
# Define the model:
    Convolutional Layers:(input channels, output channels, kernel_size, stride, padding)
    MaxPool Layers:(kernel_size, stride)
    Flatten
    Dense Layer
    Activation units
    Dropout (optional)
# Define the training loop of the model:
    Epochs, train_loader, model, optimizer, loss_fn, regularizer
# Fit the model:
    img_batches within training loop and other inputs to model
# Evaluate the model:
    Validation_loader
    Calculate accuracy for train_loader
    Calculate accuracy for validation_loader
# Predict the images and visualize the results
# Save the model & Load the model

'''
## Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os

## Check the device enabled for torch - GPU (or CPU)
print(f"Check torch version: {torch.__version__}")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print(f"The device enabled: {device}")

## Get the Dataset
folder = "data"
cifar10_train = datasets.CIFAR10(folder, download = True, train = True, transform = transforms.ToTensor())
cifar10_val = datasets.CIFAR10(folder, download = True, train = False, transform = transforms.ToTensor())

## Normalize the Datset - Along 3 Channels
imgs = torch.stack([img for img, _ in cifar10_train], dim=3)    # [C, H, W, N]
imgs_flat = imgs.view(3, -1)                                    # [C, H*W*N]
mu = imgs_flat.mean(dim=1)                                      # [C]
sigma = imgs_flat.std(dim=1)                                    # [C]

# Normalized Training Dataset
cifar10_train = datasets.CIFAR10(folder,
                                 train = True,
                                 download = False,
                                 transform = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mu, sigma)
                                 ]))

# Normalized Validation Dataset
cifar10_val = datasets.CIFAR10(folder,
                               train = False,
                               download = False,
                               transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(mu, sigma)
                               ]))

## Define the CNN Model
class NeuralNet(nn.Module):
    def __init__(self, img, nclasses, nchannels=16, nhidden=32, kernel_size = 3):
        super().__init__()
        C, H, W = img.shape
        self.conv1 = nn.Conv2d(C, nchannels, kernel_size = kernel_size, padding = kernel_size//2) # CNN:1 [nchannels, H, W]
        # MaxPool:1 [nchannels, H//2, W//2]
        self.conv2 = nn.Conv2d(nchannels, nchannels//2, kernel_size = kernel_size, padding = kernel_size//2) # CNN:2 [nchannels//2, H//2, W//2]
        # MaxPool:2 [nchannels//2, H//4, W//4]
        self.flat1 = nchannels//2 * H//4 * W//4
        self.fc1 = nn.Linear(self.flat1, nhidden) # Fully Connected Layer:1 [nchannels//2 * H//4 * W//4, nhidden]
        self.fc2 = nn.Linear(nhidden, nclasses) # Fully Connected Layer:2 [nhidden, nclasses]

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2) # MaxPool:1 [nchannels, H//2, W//2]
        out = F.max_pool2d(torch.tanh(self.conv2(x)), 2) # MaxPool:2 [nchannels//2, H//4, W//4]
        out = torch.tanh(self.fc1(out.view(-1, self.flat1))) # Here 1st dimension is for BATCH SIZE
        out = self.fc2(out)
        return out      # [Batch_Size, nclasses]

## Training loop
def training(n_epochs, loader, model, optimizer, loss_fn, l2_regularizer=0.001):
    for epoch in range(n_epochs):
        loss_epoch = 0
        time_start = time.time()
        for imgs, labels in loader:

            imgs = imgs.to(device = device) # We want the training to be on GPU
            labels = labels.to(device = device)

            outputs = model(imgs)

            loss = loss_fn(outputs, labels)
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + l2_norm * l2_regularizer

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

        # Time taken per epoch for training and loss per epoch: (total loss for all batches/number of batches)
        print(f"Epoch: {epoch+1}, Time: {time.time()-time_start}, Loss: {loss_epoch/len(loader)}")    

## Training the Dataset
img, _ = cifar10_train[0]
img = img.to(device=device)

# Turn the dataset in to batches (PyTorch requires the input data to be in batches)
train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=64, shuffle = True)
model = NeuralNet(img, nclasses=10).to(device=device) # img requires to be of dimension[C,H,W]
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-2)

training(
    n_epochs=50,
    loader = train_loader,
    model = model,
    optimizer = optimizer,
    loss_fn = loss_fn
)

## Define accuracy (Final accuracy after all epochs)
def accuracy(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)
            correct += int((predicted==labels).sum())
            total += labels.shape[0]    # labels: [batch_size, n_classes]
    accuracy = correct/total
    return accuracy

## Calculate accuracy
val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size = 64, shuffle = False)
val_accuracy = accuracy(model, val_loader)
train_accuracy = accuracy(model, train_loader)
print(f"Training Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}") 

## Visualize the results
with torch.no_grad():
    imgs, labels = next(iter(val_loader))
    imgs = imgs.to(device = device) 
    labels = labels.to(device = device)
    outputs = model(imgs)
    _, predicted = torch.max(outputs, dim=1)

# Class names for CIFAR10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

fig = plt.figure(figsize=(10,10))
for i in range(min(16, len(imgs))):
    ax = fig.add_subplot(4, 4, i+1, xticks = [], yticks = [])
    img = imgs[i].permute(1,2,0)*sigma.to(device=device) + mu.to(device=device)
    ax.imshow(img.cpu())
    color = "green" if predicted[i]==labels[i] else "red"
    ax.set_title(f"Actual: {class_names[labels[i]]}, Predicted: {class_names[predicted[i]]}")

plt.tight_layout()
plt.show()

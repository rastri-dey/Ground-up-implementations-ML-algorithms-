'''
ML Algorithm: Multilayer Perceptron
Deep Learning Framework: Pytorch
Dataset: CIFAR10

Steps:
# Import Libraries
# Get dataset - train loader, validation loader from data loader
# Define the model
# Define the Optimizer, Loss fun, epochs
# Compile the model for all epochs and for all batches:
    # Fit the model - Get the ouput (logits) for the current batch, current epoch
    # Get the loss
    # Zeros the gradient of model parameters
    # Backpropagation (calculate backward loss)
    # Update weights using Optimizer
# Predict the model output with test data
'''

## Import Libraries

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

## Get the Dataset

folder = "data"
# Training dataset
cifar10 = datasets.CIFAR10(folder, train = True, download = True, transform = transforms.ToTensor()) 
# Validation dataset
cifar10_val = datasets.CIFAR10(folder, train = False, download = True, transform = transforms.ToTensor())

'''
The CIFAR10 dataset as provided by torchvision is already distributed as training set and test/validation set
by its creators. Enabling train = True/ False doesn't create separate physical copies on the disk, it only allows
for the pre-defined training set and test/validation set to download from datasets.cifar10.
'''

## Normalize the dataset

# Stack the dataset
imgs = torch.stack([img for img,_ in cifar10], dim=3)
# Each img in cifar10: img.shape = torch.Size([3, 32, 32]). There are 50000 images in the folder
# imgs.shape = torch.size([3,32,32,50000])

# Flatten the imgs with a new shape: [C, H*W*N]
imgs_flat = imgs.view(3, -1)

# Calculate the mean along 3 channels
mu = imgs_flat.mean(dim=1)

# Calculated the Standard Deviation along the 3 channels
sigma = imgs_flat.std(dim=1)

cifar10 = datasets.CIFAR10(folder,
                           train = True,
                           download  = False,
                           transform = transforms. Compose(
                               [
                                   transforms.ToTensor(),
                                   transforms.Normalize(mu, sigma)
                               ]
                           ))

cifar10_val = datasets.CIFAR10(folder,
                               train = False,
                               download = False,
                               transform = transforms.Compose(
                                   [
                                       transforms.ToTensor(),
                                       transforms.Normalize(mu, sigma)
                                   ]
                               ))

# Load the images in batches for training & validation
train_loader = torch.utils.data.DataLoader(cifar10, batch_size = 64, shuffle = False) 
val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size = 64, shuffle = False)

# Evaluate the trained model 
def accuracy(model, loader):
    correct = 0
    total = 0
    # No gradient calculation, just outputs is sufficient
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs.view(imgs.shape[0], -1)) # input features: torch.size([64, 3*32*32]) imgs.shape = [64, 3, 32, 32]
            _, predicted = torch.max(outputs, dim=1) 
            '''
            outputs is a tensor of batch_size and classes
            no. of rows in outputs = batch_size (64 in this case)
            no. of cols in outputs = classes (10 in this case)
            dim = 1 specifes find the max along cols (classes)
            torch.max(outputs, dim=1) would return the max logit and the max index for batch_size
            64 rows of these -> [logit, class] for one image
            '''
            correct += int((labels==predicted).sum())
            total += labels.shape[0]
    
    return correct/total

## Define the model

class MLP(nn.Module):
    def __init__(self, ninputs, nhidden, nclasses):
        super().__init__()
        self.fc1 = nn.Linear(ninputs, nhidden)
        self.fc2 = nn.Linear(nhidden, nclasses)

    def forward(self, x):
        out = F.tanh(self.fc1(x)) # 1st Layer add tanh as the activation function
        out = self.fc2(out) 
        return out # logits
    
## Training the model

n_epochs = 30
learning_rate = 1e-2

img, _ = cifar10[0]
# Take all the pixels, channels as features
ninputs = len(img.view(-1))
# No. of activation units
nhidden = 512
# No. of classes
nclasses = 10

# Initialize one instance of MLP class for one image: ninputs is feature of one image
model = MLP(ninputs, nhidden, nclasses)

# Optimizer : SGD
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Loss function
loss_fn = nn.CrossEntropyLoss()

'''
Losses: https://docs.pytorch.org/docs/stable/nn.html#loss-functions
Optimizers: https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer
'''

for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        # Here ninputs expects A 2D tensor [batch_size, H*W*C]
        outputs = model(imgs.view(imgs.shape[0],-1))
        # Calculate the loss
        loss = loss_fn(outputs, labels)
        # Zeros the gradients from the previous backward pass
        optimizer.zero_grad()
        # Calculate the gradient of loss wrt each parameter (weights)
        loss.backward()
        # Update the parameters (weights)
        optimizer.step()

    acc_val = accuracy(model, val_loader)
    print(f"Epoch: {epoch} Batch_loss: {loss} Accuracy: {acc_val}")

## Predict the model output on test data
test_images, test_label = next(iter(val_loader))

with torch.no_grad:
    outputs = model(test_images.view(test_images.shape[0],-1))
    _, predict_label = torch.max(outputs, dim=1)

# Class names for CIFAR10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

fig = plt.figure(figsize=(10,10))
for i in range(min(16, len(test_images))):
    ax = fig.add_subplot(4, 4, i+1, xticks = [], yticks = [])
    # Unnormalize the images
    # Matplotlib accepts the images in format: [H, W, C]
    img = test_images[i].permute(1,2,0) * sigma + mu
    ax.imshow(img)
    color = "green" if predict_label[i]==test_label[i] else "red"
    ax.set_title(f"Predict: {class_names[predict_label[i]]}, Actual: {class_names[test_label[i]]}", color = color)

plt.tight_layout()
plt.show()

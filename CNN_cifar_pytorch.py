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
## Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

## Check the enabled device 
print(f"Pytorch version: {torch.__version__}") # Check the torch version

# Checks if a CUDA-capable GPU is detected and accessible by PyTorch
if torch.cuda.is_available():
    device = torch.device("cuda:0") # Computations to be performed on GPU
else:
    device = torch.device("cpu") # Computations to be performed on CPU
print(device)
'''
"cuda:0" indicates first GPU in a system of GPUs
'''
## Get the dataset


'''
ML Algorithm: Deep Convolutional Neural Network 
Deep Learning Framework: Tensorflow
Dataset: MNIST

Steps:
# Import Libraries
# Get Dataset
# Define the model:
    Convolutional Layers, Size of Input image, filters to use
    Activations, Maxpooling, Batch Normalization (if needed)
# Compile the model:
    Optimizer, Loss Fn, Metrics
# Fit the model:
    Trained data, epochs
# Evaluate the model:
    Calculate the training loss, accuracy
    Calculate the test loss, accuracy
# Predict the test images and visualize output
'''

## Import Libraries
import tensorflow
from tensorflow import keras
from tensorflow.keras import models, datasets, layers
import time
import matplotlib.pyplot as plt

## Get Dataset - MNIST
mnist = datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape, test_images.shape)

## Define the Model
'''
CNN: https://keras.io/api/layers/convolution_layers/convolution2d/

Convolutional Layers require: No. of Filters, Filter size - Everything else has a default value
Activation units (default is None) 

We are using Convolutional 2D for spatial image data of (H*W) as dimensions
Conv2D allows convolution over only H, W of images
With 3D images, the filter size implicitly considers a depth equivalent to image depth 
Kernel height is always the image channel dimension
Convolutional 1D is for time-series data for example and
Convolutional 3D is for video images, where convolution happens over time in 
addition to the spatial (H*W) dimensions
ReLu helps in gradient calculation during Backpropagation
Softmax in the final layer helps in giving a probability distribution over all the classes
where the probabilities sum upto 1, this is a very interpretable output
'''
model = models.Sequential(
    [
        layers.Conv2D(32, (3,3), activation = "relu", input_shape = (28,28,1)), # Conv Layer has 32 filters of (3*3)
        layers.MaxPooling2D((2,2)), # Max Pooling layer of size (2,2)
        layers.Conv2D(64, (3,3), activation = "relu"), 
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation  = "relu"),
        layers.Conv2D(64, (3,3), activation = "relu"),
        layers.Flatten(), # Convert to a single dimension
        layers.Dense(64, activation = "relu"), # Dense Layer with 64 activation units
        layers.Dense(10, activation = "softmax") # Dense Layer with 10 output classes
    ]
)
model.summary()

## Compile the Model
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
'''
# If in metrics, we do not add accuracy then only loss will be computed through the loss fn
# By explicitly mentioning accuracy, we can calculate accuracy along with loss through 
# model.evaluate() method
'''

## Model Fitting, Model Evaluation & Model predictions

n_epochs = 2
for epoch in range(n_epochs):
    time_start = time.time()

    # Fit the model with training images
    model.fit(train_images, train_labels, epochs = 1)
    print("Time spent on training:{}".format(time.time() - time_start))

    # Calculate the training loss & training accuracy
    train_loss, train_acc = model.evaluate(train_images, train_labels)
    # Calculate the test loss & test accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    # Predict the test images
    predictions = model.predict(test_images)
    print(f"Epoch = {epoch}, Train accuracy = {train_acc}, Test Accuracy = {test_acc}")

print(predictions.shape)


## Predict and visulaize the final outputs over test images 

fig = plt.figure(figsize=(10,10))
for i in range(min(16, len(test_images))):
    ax = fig.add_subplot(4, 4, i+1, xticks = [], yticks = [])
    img = test_images[i]
    ax.imshow(img, cmap='gray')
    predict_label = predictions[i].argmax()
    color = "green" if predict_label == test_labels[i] else "red"
    ax.set_title(f"Predict: {predict_label}, Actual: {test_labels[i]}", color=color) 

plt.tight_layout()
plt.show() 
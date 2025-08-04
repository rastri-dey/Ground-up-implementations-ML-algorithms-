'''
ML Algorithm: Multilayer Perceptron
Deep Learning Framework: Tensorflow
Dataset: MNIST

Steps:
# Import Libraries
# Get dataset - train, validation data
# Define the model (Layers, activation)
# Compile the model (Optimizer, Loss function)
# Fit the model (trained data, epochs)
# Calculate the accuracy for train data, test data
# Predict the model (test data)
'''

## Import the libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

## Get the dataset - MNIST
mnist = keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
'''
A tuple of 2 tuples, each tuple containing 2 NumPy arrays
'''
## Define the model

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation = tf.nn.relu),
        keras.layers.Dense(128, activation = tf.nn.relu),
        keras.layers.Dense(10, activation = tf.nn.softmax)
    ]
)
model.summary()

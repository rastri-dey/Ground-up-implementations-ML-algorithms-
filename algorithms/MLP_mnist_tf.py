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
# Predict the model output (test data)
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
        keras.layers.Flatten(input_shape = (28,28)),
        keras.layers.Dense(128, activation = tf.nn.relu),
        keras.layers.Dense(128, activation = tf.nn.relu),
        keras.layers.Dense(10, activation = tf.nn.softmax)
    ]
)

model.summary()

## Compile the model 
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
'''
Losses: https://www.tensorflow.org/api_docs/python/tf/keras/losses
Optimizers: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
'''

## Model fitting
model.fit(train_images, train_labels, epochs = 5)

## Evaluate the model
# Calculate the accuracy of trained data
train_acc, train_loss = model.evaluate(train_images, train_labels)
print(f"Training Accuracy: {train_acc}")

# Calculate the accuracy of test data
test_acc, test_loss = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")

## Predict the test data
predict_labels = model.predict(test_images)

fig = plt.figure(figsize=(10,10))
for i in range(min(16, len(test_images))):
    predict_label = predict_labels[i].argmax()
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    ax.imshow(test_images[i], cmap = plt.cm.binary)
    color = "green" if predict_label == test_labels[i] else "red"
    ax.set_title(f"Predict: {predict_label}, Actual: {test_labels[i]}", color=color)

plt.tight_layout()
plt.show()

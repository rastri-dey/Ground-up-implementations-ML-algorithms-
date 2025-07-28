## MLP Tensorflow
# Define the model (Layers)
# Compile the model (Optimizer, loss function)
# Fit the model (trained data, epochs)
# Calculate the accuracy for train data, test data
# Predict the model (test data)

## MLP Pytorch
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
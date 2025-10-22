# Ground-up implementations of ML algorithms 
This is the collection of machine learning algorithms using deep learning frameworks like PyTorch and Tensorflow that seamlessly integrate with GPUs and other accelerators to train large models on massive datasets, in a reasonable amount of time.

| # | Algorithm | Framework | Dataset | Github | Colab | Train Time | Accuracy | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Multilayer Perceptron | PyTorch | CIFAR10 | [Github](algorithms/MLP_cifar_pytorch.py) | [Colab](notebooks/MLP_cifar_pytorch.ipynb) |14 mins|0.46|
| 2 | Multilayer Perceptron | Tensorflow | MNIST | [Github](algorithms/MLP_mnist_tf.py)  | [Colab](notebooks/MLP_mnist_tf.ipynb) |22 s|0.96|
| 3 | Convolutional Neural Network | PyTorch | CIFAR10 | [Github](algorithms/CNN_cifar_pytorch.py)  | [Colab](notebooks/CNN_cifar_pytorch.ipynb) |22 mins|0.63|Base reference for all PyTorch Models |
| 4 | Convolutional Neural Network | PyTorch | MNIST |  | [Colab](notebooks/CNN_mnist_pytorch.ipynb) |2 mins |0.96 |Reuse the Pytorch CIFAR10 CNN model to train on MNIST  |
| 5 | Convolutional Neural Network | Tensorflow | MNIST | [Github](algorithms/CNN_mnist_tf.py)  | [Colab](notebooks/CNN_mnist_tf.ipynb) |12 s|0.98|  
| 6 | Recurrent Neural Network | From Scratch| Book by <br> H.G. Wells "The Time Machine" |  | [Colab](notebooks/RNN_scratch_pytorch.ipynb) | 20 mins |Perplexity = 1.2 |Pure Python implementation |
| 7 | Recurrent Neural Network | PyTorch |Book by <br> H.G. Wells "The Time Machine" | | [Colab](notebooks/RNN_pytorch.ipynb) |3 mins|Perplexity = 1.1 | |
| 8 | Transformers | PyTorch |Book by <br> H.G. Wells "The Time Machine" | | [Colab] ||||
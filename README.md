# Ground-up implementations of ML algorithms 
This is the collection of machine learning algorithms using deep learning frameworks like PyTorch and Tensorflow that seamlessly integrate with GPUs and other accelerators to train large models on massive datasets, in a reasonable amount of time.

| # | Algorithm | Framework | Dataset | Github | Colab | Train Time | Accuracy | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Multilayer Perceptron | PyTorch | CIFAR10 | [Github](algorithms/MLP_cifar_pytorch.py) | [Colab](notebooks/MLP_cifar_pytorch.ipynb) |
| 2 | Multilayer Perceptron | Tensorflow | MNIST | [Github](algorithms/MLP_mnist_tf.py)  | [Colab](notebooks/MLP_mnist_tf.ipynb) | 
| 3 | CNN | PyTorch | CIFAR10 | [Github](algorithms/CNN_cifar_pytorch.py)  | [Colab](notebooks/CNN_cifar_pytorch.ipynb) | | |Perfect base reference for all PyTorch Models | | 67% |
| 4 | CNN | PyTorch | MNIST |  | [Colab](notebooks/CNN_mnist_pytorch.ipynb) | | |Reuse the Pytorch CNN model (prepared for CIFAR10) to train on MNIST  |
| 5 | CNN | Tensorflow | MNIST | [Github](algorithms/CNN_mnist_tf.py)  | [Colab](notebooks/CNN_mnist_tf.ipynb) |  
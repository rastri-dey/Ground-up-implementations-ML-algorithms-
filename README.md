# Ground-up implementations of ML algorithms 

## Overview

This project contains a curated collection of machine learning and deep learning algorithm implementations that are written from scratch (i.e., minimal dependencies), intended for learning, experimentation and research. The implementations are designed to work with popular frameworks like PyTorch and TensorFlow, along with GPU‑compatible training scripts and datasets.

## Project Structure
```
Ground-up-implementations-ML-algorithms-/
├── algorithms/                         # Core algorithm implementations
│   ├── MLP_cifar_pytorch.py            # MLP using PyTorch
│   ├── MLP_mnist_tf.py                 # MLP using TensorFlow
│   ├── CNN_cifar_pytorch.py            # CNN using PyTorch
│   ├── CNN_mnist_tf.py                 # CNN using TensorFlow
├── dataset/                            # Dataset loader
├── notebooks/                          # Jupyter notebooks demonstration
│   ├── MLP_cifar_pytorch.ipynb         # MLP using PyTorch
│   ├── MLP_mnist_tf.ipynb              # MLP using TensorFlow
│   ├── CNN_cifar_pytorch.ipynb         # CNN using PyTorch
│   ├── CNN_mnist_tf.ipynb              # CNN using TensorFlow
│   ├── RNN_cifar_pytorch.ipynb         # RNN using PyTorch
│   ├── RNN_mnist_tf.ipynb              # RNN using TensorFlow
└── LLM_scratch.ipynb                   # LLM using PyTorch
├── LICENSE                             # MIT License
├── README.md                          
```
## Featured Algorithms

| # | Algorithm | Framework | Dataset | Code | Train Time | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Multilayer Perceptron | PyTorch | CIFAR10 | [Github](algorithms/MLP_cifar_pytorch.py) [Colab](notebooks/MLP_cifar_pytorch.ipynb) |14 min|0.46|
| 2 | Multilayer Perceptron | Tensorflow | MNIST | [Github](algorithms/MLP_mnist_tf.py) [Colab](notebooks/MLP_mnist_tf.ipynb) |22 s|0.96|
| 3 | Convolutional Neural Network | PyTorch | CIFAR10 | [Github](algorithms/CNN_cifar_pytorch.py) [Colab](notebooks/CNN_cifar_pytorch.ipynb) |22 min|0.63|
| 4 | Convolutional Neural Network | PyTorch | MNIST | [Colab](notebooks/CNN_mnist_pytorch.ipynb) |2 min |0.96 |
| 5 | Convolutional Neural Network | Tensorflow | MNIST | [Github](algorithms/CNN_mnist_tf.py) [Colab](notebooks/CNN_mnist_tf.ipynb) |12 s|0.98|
| 6 | Recurrent Neural Network | From Scratch| Book by <br> H.G. Wells "The Time Machine" |[Colab](notebooks/RNN_scratch_pytorch.ipynb) | 20 min |Perplexity = 1.2 |
| 7 | Recurrent Neural Network | PyTorch |Book by <br> H.G. Wells "The Time Machine" |[Colab](notebooks/RNN_pytorch.ipynb) |3 min|Perplexity = 1.1|
| 8 | Large Language Model (Using Generative Pre-Trained Transformers) | PyTorch |[Edith Wharton's <br> "The Verdict"](https://en.wikisource.org/wiki/The_Verdict)|[Colab](LLM_scratch.ipynb)|28 s| Perplexity = 1.9|

## Code Usage

1. Clone the repository:

```
git clone https://github.com/rastri-dey/Ground-up-implementations-ML-algorithms-.git
cd Ground-up-implementations-ML-algorithms-
```
2. Install dependencies:

```
pip install -r requirements.txt
```
3. Run a model script or notebook:

```
python -m algorithms.<algorithm_script>
```
Note: Some models include Google Colab links to launch with minimal setup.

## Contribution

We welcome contributions that:
* Improve algorithm correctness or efficiency
* Add new models or datasets
* Provide notebooks with visual explanations
* Improve documentation

Just open a Pull Request or Issue and we’ll help you get started!

## License

This project is licensed under the [MIT License](LICENSE).
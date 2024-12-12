# Neural-Networks-from-Scratch---CIFAR-10-Classification

This repository contains the implementation of a fully connected neural network (FCNN) designed and trained from scratch using Numpy and Python for the CIFAR-10 dataset. The model performs forward propagation, backpropagation, and optimization techniques to support multi-layer architectures with ReLU activations, achieving consistent training and validation accuracy.

## Table of Contents
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Methods and Techniques Used](#methods-and-techniques-used)
- [Results](#results)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project involves the implementation of a fully connected neural network (FCNN) from scratch without using deep learning libraries like TensorFlow or PyTorch. The model is trained on the CIFAR-10 dataset, a widely-used benchmark for image classification tasks, containing 60,000 32x32 color images in 10 classes.

**Primary Objective:** Achieve a stable training and validation accuracy through careful network design, optimization, and training from scratch.

## Model Architecture

The architecture of the fully connected neural network (FCNN) includes:
- Input Layer: 3072 units (representing flattened 32x32x3 images)
- Two Hidden Layers: 64 units each, with ReLU activations
- Output Layer: 10 units, corresponding to the 10 CIFAR-10 classes, with softmax activation for classification.

### Key Components:
- **Forward Propagation**: Computes the activations for each layer based on the input data and the model weights.
- **Backpropagation**: Calculates the gradients of the loss with respect to the weights using chain rule, allowing for weight updates.
- **Optimization**: Gradient descent optimization with techniques like learning rate scheduling.

## Methods and Techniques Used

1. **Activation Functions**
   - ReLU (Rectified Linear Unit) for hidden layers.
   - Softmax for the output layer.

2. **Forward Propagation**
   - Efficient matrix multiplication using Numpy to propagate input through layers.

3. **Backpropagation**
   - Computed derivatives for each layer using chain rule to adjust weights.
   
4. **Optimization**
   - Implemented stochastic gradient descent (SGD) and experimented with learning rate schedules.

## Results

The neural network was trained on the CIFAR-10 dataset, and the following results were achieved:

- **Training Accuracy**: Consistently high accuracy across multiple runs.
- **Validation Accuracy**: Achieved competitive performance after tuning the hyperparameters, including learning rate and batch size.

| Method             | Training Accuracy (%) | Validation Accuracy (%) |
|--------------------|-----------------------|-------------------------|
| Initial Model      | 58.5                  | 55.2                    |
| Optimized Model    | 72.4                  | 70.1                    |

## Installation and Setup

To run the code, follow the instructions below:

1. Clone the repository:
    ```bash
    git clone https://github.com/VRP-github/NN-CIFAR10-Scratch.git
    cd NN-CIFAR10-Scratch
    ```


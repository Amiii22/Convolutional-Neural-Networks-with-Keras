# Convolutional Neural Networks (CNN) With Keras

This project demonstrates the application of Convolutional Neural Networks (CNNs) for classifying handwritten digits from the MNIST dataset. The project includes the creation, training, and evaluation of both a **single-layer CNN** and a **two-layer CNN**. The goal is to predict the digits (0-9) based on images from the MNIST dataset.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Model Architecture](#model-architecture)
3. [Results](#results)
4. [Dependencies](#dependencies)

---

## Project Overview

The MNIST dataset is a classic dataset in machine learning that consists of 28x28 pixel grayscale images of handwritten digits (0-9). In this project, we utilize **Convolutional Neural Networks (CNNs)**, which are widely used for image classification tasks. This project includes two different CNN architectures:

1. **Single-Layer CNN**: A basic CNN model with one convolutional layer.
2. **Two-Layer CNN**: An extended model with two convolutional layers to improve accuracy.

The project uses **Keras** with **TensorFlow** backend for building and training the models.

---

## Model Architecture

### Single-Layer CNN:
- **Conv2D Layer**: 16 filters, kernel size (5, 5), activation function: ReLU
- **MaxPooling2D Layer**: Pool size (2, 2)
- **Flatten Layer**: Flattens the 2D matrix into 1D
- **Dense Layer**: 100 neurons, activation function: ReLU
- **Output Layer**: Softmax activation to predict a probability distribution for 10 classes (0-9)

### Two-Layer CNN:
- **Conv2D Layer 1**: 16 filters, kernel size (5, 5), activation function: ReLU
- **MaxPooling2D Layer 1**: Pool size (2, 2)
- **Conv2D Layer 2**: 8 filters, kernel size (2, 2), activation function: ReLU
- **MaxPooling2D Layer 2**: Pool size (2, 2)
- **Flatten Layer**: Flattens the 2D matrix into 1D
- **Dense Layer**: 100 neurons, activation function: ReLU
- **Output Layer**: Softmax activation to predict a probability distribution for 10 classes (0-9)

---

## Results

### Single-Layer CNN:
- **Test Accuracy**: 99%
- **Test Error Rate**: 1.40%

### Two-Layer CNN:
- **Test Accuracy**: 99%
- **Test Error Rate**: 1.22%

The two-layer CNN provided a slightly lower error rate compared to the single-layer model, indicating an improvement in classification performance with the added layer of convolution.
![Model Accuracy & Loss Curve](https://github.com/Amiii22/Convolutional-Neural-Networks-with-Keras/blob/master/Screenshot%202025-01-10%20163139.png) 

![Model Accuracy & Loss Curve](https://github.com/Amiii22/Convolutional-Neural-Networks-with-Keras/blob/master/Screenshot%202025-01-10%20163154.png)

---

## Dependencies

This project requires the following Python packages:

- **TensorFlow** >= 2.0
- **Keras** >= 2.0
- **NumPy** >= 1.18
- **Matplotlib** >= 3.0
- **Seaborn** >= 0.9
- **Scikit-learn** >= 0.22




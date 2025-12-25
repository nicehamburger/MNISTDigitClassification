# MNIST Digit Classification with Convolutional NN

This repository contains a complete implementation of a Convolutional Neural Network (CNN) designed to classify handwritten digits from the MNIST dataset. The project implements model selection through stratified cross-validation and hyperparameter tuning.

## Project Overview

The goal of this project is to build a robust image classifier capable of recognizing handwritten digits (0–9). By utilizing a LeNet-inspired architecture and modern optimization techniques, the final model achieves high accuracy on unseen data.

**Key Features:**
- **Data Preprocessing:** Grayscale normalization and 4D tensor reshaping for CNN compatibility.
- **Dynamic Model Creation:** Flexible CNN architecture with adjustable filters, dense units, and learning rates.
- **Hyperparameter Search:** Systematic testing of filter combinations and learning rates.
- **5-Fold Stratified Cross-Validation:** Ensures model stability and prevents overfitting by maintaining class proportions across folds.
- **Final Training:** High-performance training on the full 60,000-image dataset.

## Model Architecture

The network follows a sequential structure optimized for spatial feature extraction:

- **Convolutional Layer 1:** 64 filters (5x5), ReLU activation. Captures basic edges and corners.
- **Max Pooling 1:** 2x2 pool size. Reduces spatial dimensions while retaining critical features.
- **Convolutional Layer 2:** 128 filters (5x5), ReLU activation. Captures complex shapes like loops and intersections.
- **Max Pooling 2:** 2x2 pool size.
- **Flatten Layer:** Converts 2D feature maps into a 1D vector.
- **Dense Layer:** 120 units, ReLU activation.
- **Output Layer:** 10 units (one for each digit), Softmax activation for probability distribution.

## Hyperparameter Tuning & Results

To find the optimal configuration, a grid search was performed across the following parameters:

- **Filter Sets:** (16, 32), (32, 64), (64, 128)
- **Learning Rates:** 0.001, 0.0005

**Final Configuration:**
- Filter Combination: 64 (Layer 1) & 128 (Layer 2)
- Learning Rate: 0.0005
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

**Performance:**

| Metric                         | Result      |
|--------------------------------|------------|
| Validation Accuracy (Avg 5-Fold) | ~99%+      |
| Final Test Accuracy (Unseen Data)| 99.14%    |

## Requirements & Installation

To run this notebook, you need Python 3.x and the following libraries:

```bash
pip install numpy tensorflow scikit-learn

# Classifiers Branch

This branch contains the classifier models and related assets used for **Inception Score (IS) evaluation** of generative models, including Variational Autoencoders (VAEs) and Denoising Diffusion Probabilistic Models (DDPMs).

## Branch Structure

classifiers/
├── train-classifers-for-inceptionscore.ipynb                 # Notebook for training classifiers
├── checkpoints/ # Trained classifier weights
│ ├── mnist_best_classifier.pt
│ ├── cifar10_best_classifier.pt
│ └── flowers_best_classifier.pt
└── README.md                   # This file


## Purpose

The classifiers trained in this branch are used to compute the **Inception Score** of generated images from VAE and DDPM models. The Inception Score uses the predicted class probabilities \(p(y|x)\) of generated images to evaluate:

- **Confidence**: Are the generated images recognized as clear, distinct classes?  
- **Diversity**: Are the generated images spread across all classes of the dataset?  

## Notebooks

- **`train-classifers-for-inceptionscore.ipynb`**:  
  Demonstrates the training of ResNet-18 classifiers on the following datasets:
  - **MNIST** – 28x28 grayscale digits 
  - **CIFAR10** – 32x32 RGB images, 10 classes  
  - **Oxford Flowers** – RGB images, 102 classes  

  The notebook includes:
  - Loading and preprocessing the dataset of your choice from the mentioned datasets  
  - Training ResNet-18 models  
  - Saving trained checkpoints  

## Checkpoints

The `checkpoints/` directory contains the trained models:

- **`mnist_best_classifier.pt`** – MNIST classifier  
- **`cifar10_best_classifier.pt`** – CIFAR10 classifier  
- **`flowers_best_classifier.pt`** – Oxford Flowers classifier  

These checkpoints are used when calculating Inception Scores for generated images in VAE and DDPM experiments.

## Usage


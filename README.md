# Classifiers Branch

This branch contains the classifier models and related assets used for **Inception Score (IS) evaluation** of generative models, including Variational Autoencoders (VAEs) and Denoising Diffusion Probabilistic Models (DDPMs).


## Branch Structure


```bash
classifiers/
├── train-classifers-for-inceptionscore.ipynb                 # Notebook for training classifiers
├── checkpoints/ # Trained classifier weights
│ ├── mnist_best_classifier.pt
│ ├── cifar10_best_classifier.pt
│ └── flowers_best_classifier.pt
└── README.md                   # This file
```


## Purpose

The classifiers trained in this branch are used to compute the **Inception Score** of generated images from VAE and DDPM models. The Inception Score uses the predicted class probabilities \(p(y|x)\) of generated images to evaluate:

- **Confidence**: Are the generated images recognized as clear, distinct classes?  
- **Diversity**: Are the generated images spread across all classes of the dataset?



## How It Works (Visual Overview)

Generated Images --> Classifier Model --> Class Probabilities (p(y|x)) --> Compute IS: confidence & diversity

- Generated images from VAE/DDPM are fed into the trained classifier.  
- The classifier outputs a probability distribution over classes for each image.  
- These probabilities are used to compute the **Inception Score**, measuring:
  - **Confidence**: Are the images clearly recognized as one class?  
  - **Diversity**: Are the images spread across all possible classes?
    

## Notebooks

- **`train-classifers-for-inceptionscore.ipynb`**:  
  Demonstrates the training of ResNet-18 classifiers on the following datasets:
  - **MNIST** – 28x28 grayscale digits, 10 classes  
  - **CIFAR10** – 32x32 RGB images, 10 classes  
  - **Oxford Flowers** – RGB images, 102 classes  

  The notebook includes:
  - Specifying one of the mentioned datasets (mnist, cifar10, flowers)
  - Loading the dataset
  - Preprocessing the dataset   
  - Training ResNet-18 models  
  - Saving trained checkpoints

**Note:** This is a **Kaggle-ready notebook** and can be run directly on Kaggle after specifying the dataset name in the notebook.


## Checkpoints

The `checkpoints/` directory contains the trained models:

- **`mnist_best_classifier.pt`** – MNIST classifier  
- **`cifar10_best_classifier.pt`** – CIFAR10 classifier  
- **`flowers_best_classifier.pt`** – Oxford Flowers classifier  

These checkpoints are used when calculating Inception Scores for generated images in VAE and DDPM experiments.


## Usage

You can either:

1. **Run directly on Kaggle**  
   - Upload the notebook to Kaggle.  
   - For best performance, enable a GPU in the notebook settings.

2. **Run locally**
   - Install [Git LFS](https://git-lfs.github.com/) to handle the large checkpoint files.  
   - Clone the repository to your local machine.  
   - Update any file paths in the notebook to match your local directories.  
   - Run the notebook as usual.

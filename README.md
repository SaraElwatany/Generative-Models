# Generative Models Project (VAE & DDPM)

This repository contains implementations and experiments for **deep generative models**, including:

- **Variational Autoencoders (VAEs)**
- **Denoising Diffusion Probabilistic Models (DDPMs)**

The project explores model behavior across multiple datasets and evaluates performance using metrics such as **Fréchet Inception Distance (FID)** and **Inception Score (IS)**.


## Repository Structure

The repository is organized into multiple branches, each focusing on a specific component of the project:

```bash
main                # Documentation and project overview
feat/classifiers    # Classifier training + checkpoints (for IS score)
feat/VAEs           # VAE training, experiments, and checkpoints
feat/DDPMs          # DDPM implementation and training scripts
```


## Project Overview

This project aims to:

- Implement generative models from scratch
- Study the effect of hyperparameters (e.g., β in VAEs)
- Compare performance across datasets of increasing complexity
- Evaluate generated samples using:
  * FID → measures similarity to real data distribution
  * Inception Score (IS) → measures quality and diversity



## Branches Details


* ### feat/classifiers

Contains the notebook for training classifiers used in Inception Score computation

Includes trained checkpoints for:
    * MNIST
    * CIFAR10
    * Oxford Flowers

These classifiers are not used for generation, only for evaluation


* ### feat/VAEs

Implementation and training of Variational Autoencoders

Experiments conducted with different β values to study:
    * Reconstruction vs regularization trade-off
    * Latent space behavior

Includes:
    1. Training notebook
    2. Saved checkpoints for multiple configurations



* ### feat/DDPMs

Implementation of Denoising Diffusion Probabilistic Models

Includes:
   - Training notebook with also sampling/generation pipeline



**Note:**

Due to large file sizes, DDPM checkpoints are not stored in the repository. They are available via Google Drive.



## Datasets Used

The project evaluates models on datasets with increasing complexity:

- MNIST → Simple grayscale digits
- CIFAR10 → Low-resolution natural images
- Oxford Flowers → More complex, fine-grained dataset



## Evaluation Metrics


* ### Inception Score (IS)

Uses a trained classifier to compute:
   1. Confidence → clear class predictions
   2. Diversity → variety across classes
  

* ### Fréchet Inception Distance (FID)

Measures distance between real and generated image distributions. Lower is better.



## Key Insights


* ### VAEs
  
  * Lower β → better reconstruction, less regularized latent space
  * Higher β → smoother images, stronger latent structure, but loss of details
  * Tend to produce blurry outputs due to pixel-wise loss (e.g., MSE)



* ### DDPMs

  * Generate sharper and more realistic samples compared to VAEs
  * Require more training time and computational resources


* ### Dataset Complexity
  
  * MNIST → easiest, strong structure learning
  * CIFAR10 → moderate difficulty with more variability
  * Oxford Flowers → most challenging due to fine-grained details


## How to Use


* ### Option 1: Explore by Branch

  1. git clone https://github.com/SaraElwatany/Generative-Models.git
  2. git checkout branch_name


* ### Option 2: Run on Kaggle

  1. Upload notebooks from any branch to Kaggle
  2. Enable GPU for better performance
  3. Update paths as needed


* ### Option 3: Run Locally

  1. Install Git LFS for large files:
    git lfs install
  2. Clone the repository
  3. Update dataset/checkpoint paths
  4. Run notebooks or scripts


## Notes

- Large checkpoint files are handled using Git LFS
- DDPM checkpoints are stored externally (Google Drive) due to size limits
- Classifiers are used strictly for evaluation (IS score)


## Future Work

- Improve VAE sharpness using:
   * Perceptual loss
   * VAE-GAN hybrids
- Train DDPMs for longer schedules
- Explore advanced diffusion models (e.g., DDIM, Latent Diffusion)
- Apply models to higher-resolution datasets

# Generative Models — VAE & DDPM from Scratch

> A deep learning project exploring and comparing **Variational Autoencoders (VAEs)** and **Denoising Diffusion Probabilistic Models (DDPMs)**, implemented entirely from scratch in PyTorch and trained on Kaggle GPU notebooks.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Branches](#branches)
- [Datasets](#datasets)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results Summary](#experimental-results-summary)
- [Key Insights](#key-insights)
- [How to Use](#how-to-use)
- [Future Work](#future-work)

---

## Overview

This project implements two families of deep generative models from the ground up and systematically compares their behaviour across datasets of increasing complexity:

- **VAEs** — study the effect of the β hyperparameter on reconstruction quality, latent space structure, and image diversity
- **DDPMs** — explore the iterative denoising approach and its advantages and limitations compared to VAEs

All models are evaluated using **Fréchet Inception Distance (FID)** and **Inception Score (IS)**, with dataset-specific ResNet-18 classifiers trained for the IS computation.

---

## Repository Structure

```bash
main                # Documentation and project overview
feat/classifiers    # ResNet-18 classifier training + checkpoints (for IS computation)
feat/VAEs           # VAE implementation, experiments, and checkpoints
feat/DDPMs          # DDPM implementation, training, and generation pipeline
```

---


## Branch Structure

```
main/
│
├── Generative Models_Report.pdf     # Report documenting every single step used, the configurations used, the results (generated images and samples with comments)
└── README.md                             # This File
```

---


## Branches

### `feat/classifiers`
ResNet-18 classifiers trained per-dataset, used exclusively for **Inception Score** computation during evaluation (not for generation).

Includes trained checkpoints for:
- MNIST
- CIFAR-10
- Oxford Flowers

---

### `feat/VAEs`
Full VAE implementation with a flexible β parameter to study the reconstruction vs. regularisation trade-off.

Includes:
- Training, evaluation, and sampling notebook
- Saved checkpoints for multiple β configurations across all three datasets

> 📄 See the [feat/VAEs README](https://github.com/SaraElwatany/Generative-Models/blob/feat/VAEs/README.md) for full details, results, and usage instructions.

---

### `feat/DDPMs`
Full DDPM implementation with a modular U-Net backbone (inspired by Stable Diffusion), including forward/reverse diffusion, temporal embeddings, and iterative generation.

Includes:
- Training, evaluation, and generation notebook
- Checkpoints available via Google Drive (too large for Git storage)

| Dataset | Checkpoint |
|---|---|
| MNIST | [⬇️ Google Drive](https://drive.google.com/file/d/1NCyypYBKebnIEkilcrfvbYD24TgmjzjD/view?usp=sharing) |
| CIFAR-10 | [⬇️ Google Drive](https://drive.google.com/file/d/1hygbg--YR2el-pPI0PrPNizwtqSduI4Q/view?usp=sharing) |
| Oxford Flowers | [⬇️ Google Drive](https://drive.google.com/file/d/1mwKy32fVXQYYaWOUgbYqeEj8MKzoXzTG/view?usp=sharing) |

> 📄 See the [feat/DDPMs README](https://github.com/SaraElwatany/Generative-Models/blob/feat/DDPMs/README.md) for full details, results, and usage instructions.

---

## Datasets

| Dataset | Classes | Resolution | Complexity |
|---|---|---|---|
| **MNIST** | 10 (digits 0–9) | Resized to 32×32 grayscale | ⬛ Low |
| **CIFAR-10** | 10 (vehicles, animals) | 32×32 RGB | 🟧 Medium |
| **Oxford Flowers** | 102 flower categories | Resized to 32×32 RGB | 🟥 High |

---

## Evaluation Metrics

### Fréchet Inception Distance (FID)
Measures the statistical distance between the distributions of real and generated images using features from a pretrained network.

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g}\right)$$

**Lower FID → generated images are more realistic and diverse.**

### Inception Score (IS)
Evaluates both the **quality** (confident predictions) and **diversity** (spread across classes) of generated images using a dataset-specific classifier.

$$\text{IS} = \exp\left(\mathbb{E}_x\, D_{KL}(p(y|x) \| p(y))\right)$$

**Higher IS → images are sharp and class-diverse.**

---

## Experimental Results Summary

### VAE — Best Configuration per Dataset

| Dataset | Best β | FID ↓ | Inception Score ↑ |
|---|---|---|---|
| MNIST | 1.0 | 342.085 | 5.621 ± 0.919 |
| CIFAR-10 | 0.5 | 354.650 | 2.379 ± 0.256 |
| Oxford Flowers | 1.0 | 295.922 | 1.188 ± 0.058 |

### DDPM — Results per Dataset

| Dataset | Epochs | FID ↓ | Inception Score ↑ |
|---|---|---|---|
| MNIST | 10 | **74.295** | **5.110 ± 0.794** |
| CIFAR-10 | 15 | 449.544 | 1.602 ± 0.302 |
| Oxford Flowers | 700 | 377.436 | 31.166 ± 0.030 |

### VAE vs. DDPM — Head to Head

| | VAE | DDPM |
|---|---|---|
| **Image Sharpness** | 🟧 Blurry (MSE averaging) | ✅ Sharp (iterative denoising) |
| **Training Speed** | ✅ Fast | 🟧 Slow |
| **Data Requirements** | ✅ Works on small datasets | 🟥 Data-hungry |
| **Diversity** | 🟧 Tends to repeat structures | ✅ More varied outputs |
| **Small Dataset Performance** | ✅ Decent | 🟥 Struggles |
| **Simple Dataset Performance** | 🟧 Blurry but coherent | ✅ Sharp and diverse |

---

## Key Insights

**VAEs**
- Lower β → better reconstruction, less regularised latent space, sharper edges
- Higher β → smoother outputs, stronger latent structure, loss of fine detail
- Inherently blurry due to pixel-wise MSE loss averaging over possible outputs
- Stable and practical even on small datasets like Oxford Flowers

**DDPMs**
- Produce significantly sharper and more diverse samples than VAEs when given sufficient data
- Highly data-hungry — small datasets (e.g., Oxford Flowers with ~8K images) are not enough
- MNIST results clearly demonstrate the quality ceiling DDPMs can reach with simple data
- Require substantially more compute and training time than VAEs

**Dataset Complexity**
- **MNIST** → easiest; both models learn meaningful structure, DDPM clearly wins
- **CIFAR-10** → moderate difficulty; neither model fully succeeds under constrained training
- **Oxford Flowers** → most challenging; VAE produces blurry but structured outputs; DDPM fails to learn any structure due to insufficient data

---

## How to Use

### Option 1 — Explore by Branch

```bash
git clone https://github.com/SaraElwatany/Generative-Models.git
cd Generative-Models
git checkout feat/VAEs      # or feat/DDPMs, feat/classifiers
```

### Option 2 — Run on Kaggle

| Branch | Notebook |
|---|---|
| `feat/VAEs` | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/saraaymanelwatany/vae-pytorch) |
| `feat/DDPMs` | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/saraaymanelwatany/ddpm-from-scratch) |

### Option 3 — Run Locally

```bash
# 1. Install Git LFS for large checkpoint files
git lfs install

# 2. Clone the repository
git clone https://github.com/SaraElwatany/Generative-Models.git

# 3. Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn scipy tqdm

# 4. Checkout the branch you want and run the notebook
git checkout feat/VAEs
```

---


## Future Work

- Improve VAE sharpness using perceptual loss or VAE-GAN hybrids
- Train DDPMs for longer with larger model capacity on CIFAR-10
- Explore **DDIM** (faster sampling) and **Latent Diffusion Models** (memory-efficient, higher resolution)
- Apply models to higher-resolution datasets with proper augmentation pipelines

---

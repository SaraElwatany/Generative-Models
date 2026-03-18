# Variational Autoencoders (VAEs) — From Scratch

> Branch: `feat/VAEs`  
> Part of a broader generative modelling project exploring VAEs and DDPMs across multiple image datasets, implemented entirely from scratch using PyTorch and trained on Kaggle GPU notebooks.

---

## Table of Contents

- [Overview](#overview)
- [Branch Structure](#branch-structure)
- [Datasets](#datasets)
- [Architecture](#architecture)
- [Loss Function & Beta Parameter](#loss-function--beta-parameter)
- [Training Setup](#training-setup)
- [Evaluation Metrics](#evaluation-metrics)
- [How to Use](#how-to-use)
- [Experimental Results](#experimental-results)
- [Key Takeaways](#key-takeaways)
- [Related Branches](#related-branches)

---


## Overview

This branch contains the full implementation of **Variational Autoencoders (VAEs)** built from the ground up in PyTorch. The VAE is trained on three image datasets of varying complexity — MNIST, CIFAR-10, and Oxford Flowers — and evaluated using **Fréchet Inception Distance (FID)** and **Inception Score (IS)**.

The project investigates the effect of the **β (beta) hyperparameter** on the quality, diversity, and realism of generated images, analysing the trade-off between reconstruction fidelity and latent space regularisation.

All training was conducted on **Kaggle GPU notebooks** to leverage free GPU compute.

---


## Branch Structure

```
feat/VAEs/
│
├── vae_notebook.ipynb          # Main Kaggle notebook: build, train, evaluate, sample
│
└── checkpoints/                # Saved model checkpoints for different configs
    ├── mnist/
    │   ├── mnist_beta_0.1.pth
    │   ├── mnist_beta_0.3.pth
    │   ├── mnist_beta_0.5.pth
    │   ├── mnist_beta_1.pth
    │   └── mnist_beta_5.pth
    ├── cifar10/
    │   ├── cifar10_beta_0.3.pth
    │   ├── cifar10_beta_0.5.pth
    │   └── cifar10_beta_5.pth
    └── oxford_flowers/
        ├── flowers_beta_0.3.pth
        ├── flowers_beta_0.5.pth
        ├── flowers_beta_1.pth
        └── flowers_beta_5.pth
```

---


## Datasets

| Dataset | Classes | Image Size | Notes |
|---|---|---|---|
| **MNIST** | 10 (digits 0–9) | Resized to 32×32 grayscale (originally 28×28)  | Simple, fast to train |
| **CIFAR-10** | 10 (vehicles, animals) | 32×32 RGB | Higher complexity, diverse backgrounds |
| **Oxford Flowers** | 102 flower categories | Resized to 32×32 RGB for computational efficiency | Small dataset (~8,189 images), high visual variability |

---


## Architecture

The VAE is composed of three main components:

### Encoder
Takes the input image, passes it through a series of convolutional layers, flattens the output, and maps it to two vectors: **μ (mu)** and **log σ² (logvar)** — the parameters of the approximate posterior distribution in latent space.

### Reparameterisation Trick
The latent vector **z** is sampled using:

```
z = μ + ε · exp(0.5 · logvar),    ε ~ N(0, I)
```

This allows gradients to flow through the sampling step during backpropagation.

### Decoder
Takes the latent vector **z**, reshapes it into a spatial tensor, and passes it through transposed convolutional layers to reconstruct the image to the original input dimensions.

---


## Loss Function & Beta Parameter

The VAE loss combines two terms:

```
L = Reconstruction Loss + β · KL Divergence
```

**Reconstruction Loss options:**

| Loss | Best For | Trade-off |
|---|---|---|
| `MSE` | Continuous image outputs | Tends to produce blurrier images |
| `L1` | Less sensitivity to outliers | Can produce sharper edges than MSE |
| `BCE` | Binary/normalised [0,1] outputs | Common for grayscale datasets like MNIST |

**β (Beta) controls the regularisation strength:**

- **Low β** → Model focuses on reconstruction; less constrained latent space; sharper but potentially less diverse outputs.
- **High β** → KL divergence dominates; more structured Gaussian latent space; smoother, more abstract outputs with better distributional alignment but loss of fine detail.

---


## Training Setup

| Parameter | Value / Choice |
|---|---|
| Optimiser | Adam |
| Learning Rate Scheduler | ReduceLROnPlateau |
| Early Stopping | ✅ Enabled |
| Checkpointing | ✅ Best model saved per config |
| Resume from Checkpoint | ✅ Supported |
| Default Samples at Generation | 100 |

**Why Adam?**  
Adam combines momentum and adaptive learning rates, stabilising training on deep networks. Overfitting is controlled via early stopping and dataset size rather than weight decay.

**Why ReduceLROnPlateau?**  
Dynamically adjusts the learning rate based on training progress, enabling smoother convergence and preventing overshooting of minima.

---


## Evaluation Metrics

### Fréchet Inception Distance (FID)
Measures the statistical similarity between the distribution of generated images and real images using features extracted from a pretrained InceptionV3 network.

- **Lower FID** → generated images are more realistic and diverse, closely matching the real data distribution.

### Inception Score (IS)
Evaluates both the **quality** and **diversity** of generated images using a classifier trained on each specific dataset.

- **Higher IS** → images are confidently classified (high quality) and spread across classes (high diversity).

> A **ResNet-18 classifier** was trained separately on each dataset and used to compute Inception Score, ensuring the metric reflects class-specific quality rather than generic ImageNet features.

---


## How to Use

### Option 1 — Run on Kaggle (Recommended)

Open the notebook directly on Kaggle to use free GPU compute:

> 🔗 **[Open Kaggle Notebook Playground →](https://www.kaggle.com)**  
> *(Replace this link with your actual Kaggle notebook URL)*

Steps:
1. Open the notebook on Kaggle.
2. Enable GPU: `Settings → Accelerator → GPU T4 x2` (or P100).
3. Select your dataset: set `DATASET = "mnist"` / `"cifar10"` / `"oxford_flowers"`.
4. Set your desired beta value: `BETA = 0.1` (or `0.3`, `0.5`, `1`, `5`).
5. Run all cells to train, evaluate, and generate samples.

---

### Option 2 — Run Locally

#### 1. Clone the repository and switch to this branch

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
git checkout feat/VAEs
```

#### 2. Install dependencies

```bash
pip install torch torchvision numpy matplotlib scikit-learn scipy tqdm
```

#### 3. Configure training parameters

At the top of the notebook (or in the config cell), set:

```python
DATASET    = "mnist"       # Options: "mnist", "cifar10", "oxford_flowers"
BETA       = 1.0           # Controls KL divergence weight
LOSS_TYPE  = "mse"         # Options: "mse", "l1", "bce"
LATENT_DIM = 128           # Dimensionality of latent space
EPOCHS     = 200
LR         = 1e-3
BATCH_SIZE = 64
```

#### 4. Train the VAE

```python
# Run the training cell
train_vae(model, dataloader, optimizer, epochs=EPOCHS, beta=BETA)
```

#### 5. Resume from a checkpoint

```python
checkpoint_path = "checkpoints/mnist/vae_mnist_beta1.pth"
model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
```

#### 6. Generate samples

```python
# Generates 100 new images by sampling from the latent Gaussian distribution
generate_samples(model, n_samples=100)
```

#### 7. Evaluate

```python
fid_score  = compute_fid(real_images, generated_images)
inc_score  = compute_inception_score(generated_images)
print(f"FID: {fid_score:.3f} | IS: {inc_score[0]:.4f} ± {inc_score[1]:.4f}")
```

---


## Experimental Results

### MNIST

| β | Epochs | Train Loss | FID ↓ | Inception Score ↑ |
|---|---|---|---|---|
| 0.1 | 164 | 108.73 | 360.502 | 4.738 ± 0.742 |
| 0.3 | 185 | 118.99 | 351.394 | 4.908 ± 0.843 |
| 0.5 | 214 | 126.19 | 346.837 | 5.254 ± 0.968 |
| **1.0** | **219** | **139.14** | **342.085** | **5.621 ± 0.919** |
| 5.0 | 113 | 195.26 | 352.915 | 5.291 ± 0.536 |

**Observations:**
- As β increases from 0.1 → 1, FID **decreases** (better distributional alignment) and IS **increases** (more diverse outputs).
- At β = 5, early stopping kicks in sooner; performance degrades slightly as the KL term overwhelms reconstruction.
- Images remain blurry across all configurations — a known limitation of MSE-based VAEs.
- The VAE tends to over-generate circular digits (0, 6, 8), showing reduced diversity compared to the DDPM baseline.

---

### CIFAR-10

| β | Epochs | Train Loss | FID ↓ | Inception Score ↑ |
|---|---|---|---|---|
| 0.3 | 194 | 109.94 | 364.580 | 2.250 ± 0.310 |
| 0.5 | 164 | 134.13 | 354.650 | 2.379 ± 0.256 |
| 5.0 | 117 | 310.75 | 362.678 | 1.350 ± 0.095 |

**Observations:**
- At moderate β (0.3–0.5), the model preserves edges and some coarse object structure, but CIFAR-10's complexity (diverse backgrounds, colours, viewpoints) makes fine detail extremely difficult to recover.
- At β = 5, the model collapses into highly regularised, smooth outputs — structurally coherent blobs with little recognisable detail.
- All generated images remain blurry due to MSE reconstruction loss.

---

### Oxford Flowers

| β | Epochs | Train Loss | FID ↓ | Inception Score ↑ |
|---|---|---|---|---|
| 0.3 | 212 | 124.50 | 302.729 | 1.141 ± 0.034 |
| 0.5 | 218 | 150.18 | 315.772 | 1.168 ± 0.035 |
| **1.0** | **163** | **197.59** | **295.922** | **1.188 ± 0.058** |
| 5.0 | 134 | 319.74 | 319.678 | 1.172 ± 0.028 |

**Observations:**
- β = 1 achieves the best FID (295.922) on this dataset — a middle ground between reconstruction fidelity and latent space structure.
- Higher β values force the latent space to be more Gaussian but sacrifice texture and background detail.
- High intra-class variability (petal shapes, colours, backgrounds across 102 categories) makes this the most challenging dataset for the VAE.

---


## Key Takeaways

- **VAEs are fast and stable to train**, even on small datasets, but are fundamentally constrained by reconstruction blurriness — particularly when using MSE loss, which averages over possible outputs.
- **The β parameter is a powerful dial**: higher β improves distributional alignment (lower FID) up to a point, after which excessive regularisation harms reconstruction quality.
- **Early stopping** is critical to prevent wasted computation and overfitting.
- **VAEs remain competitive on small datasets** where DDPMs struggle due to data hunger — see the Oxford Flowers results for a direct demonstration of this.

---


## Related Branches

| Branch | Description |
|---|---|
| `main` | Project overview, and final report |
| `feat/classifiers` | ResNet-18 classifiers trained per-dataset for Inception Score computation |
| `feat/VAEs` | ← **You are here** |
| `feat/DDPMs` | Denoising Diffusion Probabilistic Models (U-Net based), trained from scratch |

---

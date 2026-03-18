# Denoising Diffusion Probabilistic Models (DDPMs) — From Scratch

> Branch: `feat/DDPMs`  
> Part of a broader generative modelling project exploring VAEs and DDPMs across multiple image datasets, implemented entirely from scratch using PyTorch and trained on Kaggle GPU notebooks.

---

## Table of Contents

- [Overview](#overview)
- [Branch Structure](#branch-structure)
- [Datasets](#datasets)
- [Architecture](#architecture)
- [Noise Scheduler](#noise-scheduler)
- [Temporal Embeddings](#temporal-embeddings)
- [Training Setup](#training-setup)
- [Evaluation Metrics](#evaluation-metrics)
- [How to Use](#how-to-use)
- [Experimental Results](#experimental-results)
- [Key Takeaways](#key-takeaways)
- [Related Branches](#related-branches)

---

## Overview

This branch contains the full implementation of **Denoising Diffusion Probabilistic Models (DDPMs)** built from the ground up in PyTorch. The model is based on a **modular U-Net architecture** (inspired by Stable Diffusion / Hugging Face Diffusers) and trained on three image datasets of varying complexity — MNIST, CIFAR-10, and Oxford Flowers.

The DDPM learns to reverse a gradual noising process, starting from pure Gaussian noise and iteratively denoising to produce realistic image samples. Performance is evaluated using **Fréchet Inception Distance (FID)** and **Inception Score (IS)**.

All training was conducted on **Kaggle GPU notebooks** to leverage free GPU compute.

---

## Branch Structure

```
feat/DDPMs/
│
├── ddpm_notebook.ipynb          # Main Kaggle notebook: build, train, evaluate, generate
```

| Dataset | Checkpoint | Download |
|---|---|---|
| MNIST | `ddpm_mnist.pt` | [⬇️ Google Drive](https://drive.google.com/file/d/1NCyypYBKebnIEkilcrfvbYD24TgmjzjD/view?usp=sharing) |
| CIFAR-10 | `ddpm_cifar10.pt` | [⬇️ Google Drive](https://drive.google.com/file/d/1hygbg--YR2el-pPI0PrPNizwtqSduI4Q/view?usp=sharing) |
| Oxford Flowers | `ddpm_flowers.pt` | [⬇️ Google Drive](https://drive.google.com/file/d/1mwKy32fVXQYYaWOUgbYqeEj8MKzoXzTG/view?usp=sharing) |

---

## Datasets

| Dataset | Classes | Image Size | Notes |
|---|---|---|---|
| **MNIST** | 10 (digits 0–9) | 28×28 grayscale | Simple, fast to train — DDPM excels here |
| **CIFAR-10** | 10 (vehicles, animals) | 32×32 RGB | Higher complexity, requires more training |
| **Oxford Flowers** | 102 flower categories | Resized to 32×32 RGB | Small dataset (~8,189 images) — data-hungry DDPMs struggle |

---

## Architecture

The DDPM uses a **modular U-Net** with downsampling, middle, and upsampling blocks connected via skip connections — the same backbone behind modern diffusion models like Stable Diffusion.

### Downsampling Blocks
A stack of repeatable residual + attention blocks that progressively reduce spatial resolution while doubling channel count, balancing computational efficiency with feature preservation.

Each block contains:
- **Residual Block:** Group Norm → SiLU activation → Convolution → Add temporal embedding → Skip connection
- **Attention Block:** Group Norm → Multi-head self-attention (captures long-range dependencies)
- Optional downsampling at the end of each block

### Middle Block
Sits at the bottleneck of the U-Net. Applies a residual block, followed by interleaved attention and residual blocks. Output is prepared for the upsampling path.

### Upsampling Blocks
Mirror image of the downsampling path:
- Upsample spatial resolution
- Apply residual + attention blocks
- Receive skip-connected feature maps from the corresponding downsampling layer via concatenation

### Full U-Net Flow

```
Input Image (noised)
        │
   [Downsampling Blocks]  ──────────────────────┐ skip connections
        │                                        │
   [Middle Block]                                │
        │                                        │
   [Upsampling Blocks]  ←───────────────────────┘
        │
   Predicted Noise Residual
```

**Why U-Net?**
- Multi-scale feature extraction via skip connections
- Attention layers capture global image structure
- Residual connections prevent vanishing gradients during reverse diffusion
- Temporal embeddings condition the network on the current noise level

---

## Noise Scheduler

The scheduler governs both directions of the diffusion process:

**Forward Process** — gradually adds Gaussian noise to an image over `T` timesteps until the image becomes pure noise:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t \mathbf{I})$$

**Reverse Process** — the model learns to denoise step by step, predicting the noise residual at each timestep:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t))$$

The scheduler computes the mean and variance for each reverse step to ensure smooth, stable denoising rather than abrupt jumps.

---

## Temporal Embeddings

The U-Net is conditioned on the current timestep `t` using **sinusoidal positional embeddings** — the same technique used in Transformers:

$$\text{emb}(t)_i = \begin{cases} \sin(t / 10000^{2i/d}) & \text{if } i < d/2 \\ \cos(t / 10000^{2i/d}) & \text{if } i \geq d/2 \end{cases}$$

The embedding vector is projected via linear layers and added to each residual block, allowing the network to adapt its denoising behaviour to the current noise level.

---

## Training Setup

| Parameter | Value / Choice |
|---|---|
| Optimiser | Adam |
| Loss Function | MSE (predicted noise vs. true noise) |
| Learning Rate Scheduler | ReduceLROnPlateau |
| Early Stopping | ✅ Enabled |
| Checkpointing | ✅ Best model saved per dataset |
| Resume from Checkpoint | ✅ Supported |

**Training loop per batch:**
1. Sample random noise `ε ~ N(0, I)` and a random timestep `t`
2. Add noise to the image using the forward scheduler → `x_t`
3. Pass `x_t` and `t` through the U-Net to predict `ε_θ`
4. Compute MSE loss: `L = ||ε − ε_θ||²`
5. Backpropagate and update weights

---

## Evaluation Metrics

### Fréchet Inception Distance (FID)
Measures the statistical similarity between the distribution of generated images and real images.

- **Lower FID** → generated images are more realistic and diverse.

### Inception Score (IS)
Evaluates both the **quality** and **diversity** of generated images using a dataset-specific classifier.

- **Higher IS** → images are confidently classified and varied across classes.

> A **ResNet-18 classifier** was trained separately on each dataset (see `feat/classifiers` branch) and used to compute Inception Score, ensuring the metric is tailored to each dataset's class distribution.

---

## How to Use

### Option 1 — Run on Kaggle (Recommended)

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/saraaymanelwatany/ddpm-pytorch)

Steps:
1. Open the notebook on Kaggle.
2. Enable GPU: `Settings → Accelerator → GPU T4 x2` (or P100).
3. Select your dataset: set `DATASET = INDEX` where INDEX: <kbd>0</kbd> → MNIST · <kbd>1</kbd> → CIFAR-10 · <kbd>2</kbd> → Oxford Flowers
4. Configure the noise scheduler: set `T` (number of diffusion timesteps).
5. Run all cells to train, evaluate, and generate samples.

---

### Option 2 — Run Locally

#### 1. Clone the repository and switch to this branch

```bash
git clone https://github.com/SaraElwatany/Generative-Models.git
cd Generative-Models
git checkout feat/DDPMs
```

#### 2. Install dependencies

```bash
pip install torch torchvision numpy matplotlib scikit-learn scipy tqdm
```

#### 3. Configure training parameters

```python
DATASET    = 0            # 0: MNIST | 1: CIFAR-10 | 2: Oxford Flowers
T          = 1000         # Number of diffusion timesteps
EPOCHS     = 50
LR         = 1e-4
BATCH_SIZE = 64
```

#### 4. Train the DDPM

```python
train_ddpm(model, scheduler, dataloader, optimizer, epochs=EPOCHS)
```

#### 5. Resume from a checkpoint

Download the checkpoint for your dataset:

| Dataset | Download |
|---|---|
| MNIST | [⬇️ Google Drive](https://drive.google.com/file/d/1NCyypYBKebnIEkilcrfvbYD24TgmjzjD/view?usp=sharing) |
| CIFAR-10 | [⬇️ Google Drive](https://drive.google.com/file/d/1hygbg--YR2el-pPI0PrPNizwtqSduI4Q/view?usp=sharing) |
| Oxford Flowers | [⬇️ Google Drive](https://drive.google.com/file/d/1mwKy32fVXQYYaWOUgbYqeEj8MKzoXzTG/view?usp=sharing) |

```python
checkpoint_path = "checkpoints/mnist/ddpm_mnist.pt"
model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
```

#### 6. Generate samples

```python
# Starts from pure noise and iteratively denoises over T timesteps
generate_samples(model, scheduler, n_samples=100)
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

| Epochs | Train Loss | FID ↓ | Inception Score ↑ |
|---|---|---|---|
| 10 | 0.0003 | **74.295** | **5.110 ± 0.794** |

**Observations:**
- The DDPM produces sharp, well-defined digits after only 10 epochs — demonstrating its advantage over VAEs on simple datasets.
- FID of 74.295 is significantly lower than all VAE configurations (best VAE FID: 342.085), indicating the generated distribution is much closer to real MNIST.
- Images are diverse across all digit classes with no tendency to over-generate particular shapes.
- Clear boundaries and crisp strokes — a direct consequence of the iterative denoising approach rather than averaging over outputs as MSE-based VAEs do.

---

### CIFAR-10

| Epochs | Train Loss | FID ↓ | Inception Score ↑ |
|---|---|---|---|
| 15 | 0.0021 | 449.544 | 1.602 ± 0.302 |

**Observations:**
- The DDPM struggles significantly with CIFAR-10 under these training constraints.
- The high FID of 449.544 reflects that generated images lack recognisable object structure.
- Root causes: CIFAR-10's high intra-class variability, diverse textures and backgrounds, and the 32×32 resolution limiting recoverable detail during denoising.
- DDPMs are data-hungry and require extensive training — 15 epochs with 60,000 images is insufficient for the model to reliably learn the reverse diffusion process.
- Generated outputs show some colour patterns vaguely resembling the dataset distribution, but no coherent objects.
- Neither DDPM nor VAE produces satisfactory results on CIFAR-10 under these conditions, though for different reasons: VAE suffers from blurriness; DDPM suffers from insufficient training.

---

### Oxford Flowers

| Epochs | Train Loss | FID ↓ | Inception Score ↑ |
|---|---|---|---|
| 700 | 0.0028 | 377.436 | 31.166 ± 0.030 |

**Observations:**
- The worst results across all experiments — despite 700 training epochs, the model generates no recognisable flower shapes.
- Outputs are noisy, colourful patches that reflect the dataset's bright colour palette but nothing more.
- The fundamental problem is **dataset size**: only 8,189 images across 102 categories gives the model very few examples per class.
- Downscaling to 32×32 removes most fine-grained detail (petal shapes, textures), leaving insufficient visual information per sample.
- The high IS of 31.166 is misleading here — it reflects confident colour-based predictions by the classifier rather than genuine structural quality.
- **Better approaches for this dataset:** latent diffusion models (more memory-efficient, operate on compressed latent representations) or data augmentation to artificially expand the dataset.

---

## Key Takeaways

- **DDPMs produce sharper, more diverse images than VAEs** when given sufficient data and training time, as clearly demonstrated on MNIST.
- **Data quantity is critical** — DDPMs are fundamentally data-hungry. Small datasets like Oxford Flowers (8,189 images) are not well-suited for DDPM training without augmentation or architectural adaptations like latent diffusion.
- **Resolution matters** — downscaling to 32×32 to manage memory removes fine detail, limiting the model's ability to learn meaningful structure.
- **More training epochs** are needed for complex datasets; the CIFAR-10 results would likely improve substantially with 100+ epochs and a larger model.
- **Temporal embeddings** are essential — they allow the U-Net to adapt its behaviour to the noise level at each step, enabling clean iterative denoising.
- **Skip connections** in the U-Net are key to recovering fine spatial detail lost during downsampling, directly contributing to the sharpness advantage over VAEs.

---

## Related Branches

| Branch | Description |
|---|---|
| `main` | Project overview, combined results, and final report |
| `feat/classifiers` | ResNet-18 classifiers trained per-dataset for Inception Score computation |
| `feat/VAEs` | Variational Autoencoders trained from scratch with β-VAE analysis |
| `feat/DDPMs` | ← **You are here** |

---

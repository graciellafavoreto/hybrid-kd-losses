Knowledge Distillation with Hybrid Losses for Lung Nodule Malignancy Classification

This repository contains the code for the paper submitted to CBMS. It implements a knowledge distillation framework for lung nodule malignancy classification using the [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) dataset.

## Overview

Five teacher models (ResNet-50, DenseNet-121, ViT-Small, Swin-Base, MaxViT-Tiny) are trained to classify lung nodules as benign or malignant. Their knowledge is then distilled into two lightweight student models (EfficientNet-B0 and MobileNetV2) via four distillation loss variants, yielding **120 experiments** in total.

## Pipeline

**1. Preprocessing** (`preprocessing_crop_3_slices.py`)  
Queries LIDC-IDRI scans via `pylidc`, computes annotator consensus, and extracts a 128×128×3 crop (slices *k−1*, *k*, *k+1*) centered on each nodule centroid. Each slice is bicubically upscaled to 224×224. Metadata and malignancy scores are saved to `nodules_metadata.csv`.

**2. Teacher Training** (`teacher_models.py`)  
Trains the five teacher models with cross-entropy loss and class-weighted balancing. Models are selected by best validation AUC and checkpointed to `checkpoints_teachers_equals/`.

**3. Knowledge Distillation** (`student_models.py`)  
Trains student models under four distillation loss variants and three temperatures (τ ∈ {4, 10, 30}), for each teacher. The best student configuration (by val AUC) is evaluated on the test set.

## Distillation Loss Variants

| Key | Formula |
|-----|---------|
| `classic` | (1−α)SL + ατ²KL |
| `kl_mse` | classic + MSE(soft) |
| `kl_focal` | classic + Focal(soft) |
| `kl_mse_focal` | classic + MSE(soft) + Focal(soft) |

α = 0.5, Focal γ = 2.0

## Labeling Strategy

Nodules are labeled using the median malignancy score across annotators:
- **Benign (0):** median ≤ 2.5
- **Malignant (1):** median ≥ 3.5
- **Ambiguous:** excluded

## Data Split

Patient-level stratified split (no data leakage):
- Train ≈ 5/7 | Val ≈ 1/7 | Test ≈ 1/7

## Requirements

```
torch, timm, pylidc, numpy, pandas, scipy, scikit-learn, tqdm
```

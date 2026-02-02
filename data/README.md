## **DailyOCE**: Generation Guide

This guide covers the **DailyOCE** dataset - a multimodal behavioral dataset for complex event detection in smart home environments.

The dataset generation process creates multimodal (IMU + audio) embeddings for training and evaluating complex event detection models. The system supports multiple dataset variants for different experimental settings.

## Prerequisites

**Place the following folders under `data/` directory:**

- (Optional) `Audio/` and `IMU/` directories with raw sensor data (only needed for generating Multimodal/)
- Raw multimodal data in `Multimodal/` directory
- Pretrained models in `saved_models/` directory (includes AE classifier and pretrained multimodal encoders):
  - BEATs (audio encoder): `saved_models/BEATs/`
  - LIMUBert (IMU encoder): `saved_models/LIMUBert/`
  - Multimodal embedding models: `saved_models/multimodal_embed_model_w2.0.pt`, `multimodal_embed_model_w2.5.pt`
  - AE Classifiers: `saved_models/AE_classifier_w2.0.pt`, `saved_models/AE_classifier_w2.5.pt`
- NAR datasets under `CE_dataset/` directory:
  - `CE_dataset/nar/` - Standard 2.5s windows
  - `CE_dataset/nar_w2.0s/` - Realistic 2.0s windows
- GPU recommended for embedding generation

**Download**: All pretrained models and raw data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1O9odepHFZbnjSBnWIwNoU1nov4kqRxyA?usp=sharing).

## NAR Datasets (Concept Traces)

NAR (Neural Algorithmic Reasoner) datasets contain ground-truth atomic event labels as AE-level concept traces, generated using LLM-based simulators. These datasets are used for training the NAR model on symbolic reasoning patterns before adapting to raw sensor data.

**Key Characteristics:**
- **Format**: Integer tokens (0-8) representing atomic activities
- **Size**: 40,000 samples for training, 5000 for validation, 5000 for test
- **Duration**: 5-minute CE sequences only
- **Place under directories**:
  - `data/CE_dataset/nar/` - Standard 2.5s windows
  - `data/CE_dataset/nar_w2.0s/` - Realistic 2.0s windows

**Download**: NAR datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Q0xu5amYVsXi4U37uNzOdmgQbdzrksA1?usp=sharing).

## Dataset Variants

## 1. Standard (Idealized Windowing)

Perfectly aligned 2.5-second detection windows with ground truth events.

## Step 1: Generate Index Files

Index files specify which raw sensor windows to use for each sample.

**Note:** For reproducibility, the index files used in the paper are provided in the repository. You can use these directly to regenerate the exact datasets, or generate new index files for different experiments.

```bash
# Training set (specify n_samples)
python generate_indices.py --duration 5 --split train --n_samples 2000

# Validation set
python generate_indices.py --duration 5 --split val --n_samples 2000

# Test set (cross-subject)
python generate_indices.py --duration 5 --split test --n_samples 2000

# Test set (cross-subject, longer duration)
python generate_indices.py --duration 15 --split test --n_samples 2000
```

**Output:** `CE_dataset/ce5min_{split}_indices_{n}.npz`

## Step 2: Generate Embeddings

Generate 128-dim fusion embeddings from raw sensor data.

```bash
# Training set (requires --n_samples)
python dataset_loader.py --duration 5 --split train --n_samples 2000

# Validation set
python dataset_loader.py --duration 5 --split val

# Test set (cross-subject)
python dataset_loader.py --duration 5 --split test

# Test set (cross-subject, longer duration)
python dataset_loader.py --duration 15 --split test
```

**Output:**
- `CE_dataset/standard/ce5min_{split}_data_{n}.npy` (embeddings, shape: n × 120 × 128)
- `CE_dataset/standard/ce5min_{split}_labels_{n}.npy` (labels, shape: n × 120 × 10)

## 2. Shifted (Temporal Augmentation)

Applies temporal shift to raw sensor data for robustness testing.

```bash
# Generate test set with 0.1s temporal shift
python dataset_loader.py --duration 5 --split test --shift 0.1
```

**Output:** `CE_dataset/ideal_shifted/ce5min_test_data_shift0.1.npy`

## 3. Fused (Adjacent Window Fusion)

Weighted fusion of current window with portion of previous window.

```bash
# Generate test set with 10% fusion from previous window
python dataset_loader.py --duration 5 --split test --fuse 0.1
```

**Output:** `CE_dataset/ideal_fused/ce5min_test_data_fuse0.1.npy`

## 4. AE2CE (Classifier Predictions)

Uses atomic event classifier predictions (9-dim) instead of raw embeddings (128-dim).

```bash
# One-hot predictions (hard labels)
python dataset_loader.py --duration 5 --split train --n_samples 2000 --ae2ce

# Soft probabilities
python dataset_loader.py --duration 5 --split train --n_samples 2000 --soft-ae2ce
```

**Output:**
- One-hot: `CE_dataset/ae2ce/ae2ce5min_{split}_data_{n}.npy` (shape: n × 120 × 9)
- Soft: `CE_dataset/softae2ce/softae2ce5min_{split}_data_{n}.npy` (shape: n × 120 × 9)

**Notes:**
- Faster inference due to smaller input dimensions (9 vs 128)
- Cannot combine with `--shift` flag
- Requires pretrained AE classifier

## 5. Realistic Windowing

Variable detection windows that simulate real-world deployment with majority-vote atomic event aggregation.

## Generate Realistic Windowing Datasets

```bash
# Training set with 2.0s detection windows
python realistic_windowing.py \
    --dataset-type standard \
    --input-dir CE_dataset \
    --raw-data-dir Multimodal \
    --detection-window 2.0 \
    --output-dir CE_dataset/realistic_w2.0s \
    --split train \
    --n-samples 2000 \
    --gpu-id 0

# Validation set
python realistic_windowing.py \
    --dataset-type standard \
    --input-dir CE_dataset \
    --raw-data-dir Multimodal \
    --detection-window 2.0 \
    --output-dir CE_dataset/realistic_w2.0s \
    --split val \
    --gpu-id 0

# Test set
python realistic_windowing.py \
    --dataset-type standard \
    --input-dir CE_dataset \
    --raw-data-dir Multimodal \
    --detection-window 2.0 \
    --output-dir CE_dataset/realistic_w2.0s \
    --split test \
    --duration 5 \
    --gpu-id 0
```

**Output:**
- `ce5min_{split}_data_{n}.npy`: Fresh embeddings (n × n_windows × 128)
- `ce5min_{split}_ae_{n}.npy`: Noisy AE labels from majority voting
- `ce5min_{split}_labels_{n}.npy`: CE labels (n × n_windows × 10)



**Requirements:**
- Must have existing realistic windowing embeddings generated first
- Much faster than regenerating embeddings from scratch

## Dataset Durations

All commands support different CE durations via `--duration`:
- `5`: 5-minute CE sequences (120 windows at 2.5s)
- `15`: 15-minute CE sequences (360 windows at 2.5s)
- `30`: 30-minute CE sequences (720 windows at 2.5s)

## Dataset Splits

## Training Set
- Subjects: 1-4
- Variable sample sizes: 2000, 4000, 6000, 8000, 10000
- Always requires `--n_samples` parameter

## Validation Set
- Subjects: 1-4
- Fixed size, no `--n_samples` needed

## Test Sets

**Cross-Subject:**
- Subject: 5 only
- Tests generalization to new subjects
- Supports multiple durations (5min, 15min, 30min)

## File Naming Convention

**Index Files:**
- `ce{duration}min_{split}_indices_{n}.npz` (train)
- `ce{duration}min_{split}_indices.npz` (val/test)

**Embedding Files:**
- `ce{duration}min_{split}_data_{n}.npy` (embeddings)
- `ce{duration}min_{split}_labels_{n}.npy` (CE labels)

**Variants:**
- Shifted: `ce{duration}min_{split}_data_shift{value}.npy`
- Fused: `ce{duration}min_{split}_data_fuse{value}.npy`
- AE2CE: `ae2ce{duration}min_{split}_data_{n}.npy`
- Soft AE2CE: `softae2ce{duration}min_{split}_data_{n}.npy`

## Directory Structure

```
data/
├── CE_dataset/
│   ├── ce5min_*_indices_*.npz          # Index files
│   ├── standard/                        # Standard embeddings
│   │   ├── ce5min_*_data_*.npy         # 128-dim fusion embeddings
│   │   └── ce5min_*_labels_*.npy       # CE labels (10-class multi-label)
│   ├── ideal_shifted/                   # Shifted datasets
│   ├── ideal_fused/                     # Fused datasets
│   ├── ae2ce/                           # One-hot AE predictions (9-dim)
│   ├── softae2ce/                       # Soft AE predictions (9-dim)
│   ├── nar/                             # NAR datasets (concept traces, 2.5s windows)
│   │   ├── ct_ce5min_*_data_*.npy      # Ground-truth AE labels (int64)
│   │   └── ct_ce5min_*_labels_*.npy    # CE labels (10-class multi-label)
│   ├── nar_w2.0s/                       # NAR realistic windowing (2.0s windows)
│   │   ├── ct_ce5min_*_data_*.npy      # Aggregated AE labels
│   │   └── ct_ce5min_*_labels_*.npy    # CE labels
│   └── realistic_w{W}s/                 # Realistic windowing datasets
│       ├── ce5min_*_data_*.npy         # Fresh embeddings
│       ├── ce5min_*_ae_*.npy           # Noisy AE labels
│       └── ce5min_*_labels_*.npy       # CE labels
│
├── Audio/                               # Raw audio sensor data (optional, for generating Multimodal/)
│
├── IMU/                                 # Raw IMU sensor data (optional, for generating Multimodal/)
│
├── Multimodal/                          # Preprocessed and synchronized multimodal data
│   ├── MultimodalDataset_audio-40000_1234_imu-50_1234.npz  # Subjects 1-4
│   ├── MultimodalDataset_audio-40000_5_imu-50_5.npz        # Subject 5
│   ├── fusion_*_embeddings.npz         # Cached embeddings (speedup)
│   └── dataset_config_w2.5.json        # Label mappings
│
└── saved_models/                        # Pretrained models (AE classifier + multimodal encoders)
    ├── AE_classifier_w2.0.pt           # AE classifier (2.0s windows)
    ├── AE_classifier_w2.5.pt           # AE classifier (2.5s windows)
    ├── multimodal_embed_model_w2.0.pt  # Multimodal embedding model (2.0s windows)
    ├── multimodal_embed_model_w2.5.pt  # Multimodal embedding model (2.5s windows)
    ├── BEATs/                          # Audio encoder
    └── LIMUBert/                       # IMU encoder
```

## Quick Start Example

Generate a complete training dataset:

```bash
# Step 1: Generate index file (fast)
python generate_indices.py --duration 5 --split train --n_samples 2000

# Step 2: Generate embeddings (slow, GPU-accelerated)
python dataset_loader.py --duration 5 --split train --n_samples 2000

# Optional: Generate AE2CE variant (fast)
python dataset_loader.py --duration 5 --split train --n_samples 2000 --ae2ce
```

## Performance Notes

- **Index generation**: Very fast (<1 second per 1000 samples)
- **Embedding generation (standard)**: ~0.1s per sample with cached fusion embeddings
- **Embedding generation (shifted)**: ~1-2s per sample (requires raw data reconstruction)
- **AE2CE generation**: Fast (~0.03s for 7200 windows on GPU)
- **Realistic windowing**: ~30-60s per 1000 samples on GPU

## Troubleshooting

**CUDA out of memory:**
- Reduce batch size in the script
- Process fewer samples at once
- Use CPU (slower but works)

**Missing pretrained models:**
- Verify `saved_models/` directory contains BEATs, LIMUBert, and AE classifier
- Download models if missing

**Dataset shape mismatch:**
- Standard: (n_samples, 120, 128) for embeddings
- AE2CE: (n_samples, 120, 9) for predictions
- Labels: (n_samples, 120, 10) for multi-label CE
- Regenerate datasets if shapes don't match

**Cannot combine --shift with --ae2ce:**
- AE2CE uses cached embeddings; shifts require raw data reconstruction
- Generate shifted embeddings first, then apply AE classifier separately

# NAROCE: Neural Algorithmic Reasoner for Complex Event Detection

A two-stage deep learning framework for detecting complex behavioral events from multimodal sensor data (IMU + audio).

## Setup

```bash
conda env create -f requirements.yml
conda activate naroce_env
```

## Dataset Preparation

Before training models, you need to prepare the datasets. The `data/CE_dataset/` directory contains index files and datasets for complex event detection.

### Dataset Variants

**Idealized Windowing (Standard):**
- **Aligned**: Detection windows perfectly aligned with ground truth events
- **Fused**: Multimodal fusion embeddings with adjacent window fusion
- **Shifted**: Time-shifted variants for robustness testing

**Realistic Windowing:**
- Variable detection windows that simulate real-world deployment
- Location: `data/CE_dataset/realistic_w{W}s/` (e.g., `w2.0s` for 2.0-second windows)
- Uses majority-vote atomic event aggregation

**For detailed dataset generation instructions, see [`data/README.md`](data/README.md).**

## Quick Start

### Train Baseline Model
```bash
python baseline.py configs/baseline_train.json <model> <dataset_size> <seed>

# Example: Train Mamba baseline on 2000 samples
python baseline.py configs/baseline_train.json mamba1 2000 53
```

### Train NAROCE
```bash
# Stage 1: Train NAR
python naroce.py configs/naroce_nar_train.json <nar_dataset> <seed>

# Stage 2: Train Adapter (frozen NAR)
python naroce.py configs/naroce_adapter_train.json <nar_dataset> <seed> \
    --sensor_dataset <sensor_dataset> \
    --adapter_model <adapter_model>

# Stage 3: Finetune both NAR and Adapter
python naroce.py configs/naroce_finetune_train.json <nar_dataset> <seed> \
    --sensor_dataset <sensor_dataset> \
    --adapter_model <adapter_model>

# Example
python naroce.py configs/naroce_nar_train.json 20000 53
python naroce.py configs/naroce_adapter_train.json 20000 53 \
    --sensor_dataset 2000 --adapter_model mamba1_6L
```

### Evaluate Models
```bash
# Baseline
python evaluate.py configs/baseline_eval.json <model> <dataset_size> <eval_dataset> <seed>

# NAROCE (finetuned)
python evaluate.py configs/ft_naroce_eval.json <model> <dataset_size> <eval_dataset> <seed> \
    --nar_dataset <nar_dataset>

# Examples
python evaluate.py configs/baseline_eval.json mamba1 2000 15min 53
python evaluate.py configs/ft_naroce_eval.json naroce_mamba1_6L 2000 15min 53 --nar_dataset 20000
```

### Evaluation Options

**Test Dataset Durations:**
- `5min`, `15min`, `30min`: Standard test durations

**Dataset Variants:**
- **Ideal**: Perfectly aligned detection windows (default)
- **Fused**: Multimodal fusion embeddings with adjacent window fusion (`--fuse`)
- **Shifted**: Time-shifted variants for robustness testing (`--shift`)
- **Realistic**: Variable detection windows (e.g., 2.0s windows, use realistic configs)

**Examples (NAROCE):**

```bash
# Standard idealized evaluation
python evaluate.py configs/ft_naroce_eval.json naroce_mamba1_6L 2000 15min 53 --nar_dataset 40000

# Fused embeddings evaluation (0.1 fusion factor)
python evaluate.py configs/ft_naroce_eval.json naroce_mamba1_6L 2000 15min 53 --nar_dataset 40000 --fuse 0.1

# Shifted evaluation (0.1 shift factor)
python evaluate.py configs/ft_naroce_eval.json naroce_mamba1_6L 2000 15min 53 --nar_dataset 40000 --shift 0.1

# Realistic windowing evaluation (2.0s windows)
python evaluate.py configs/ft_naroce_eval_realistic_w2.0s.json naroce_mamba1_6L 2000 15min 53 --nar_dataset 40000
```

**Baseline evaluation** (same arguments, no `--nar_dataset`):
```bash
python evaluate.py configs/baseline_eval.json mamba1 2000 15min 53
```

## Configuration

All training uses JSON config files in `configs/`:
- `baseline_train.json` / `baseline_eval.json`: Baseline configuration
- `naroce_nar_train.json`: NAR training (Stage 1)
- `naroce_adapter_train.json`: Adapter training with frozen NAR (Stage 2)
- `naroce_finetune_train.json`: Joint finetuning (Stage 3)
- `ft_naroce_eval.json`: Finetuned NAROCE evaluation

Key parameters:
- `batch_size`: Default 256
- `learning_rate`: NAR=5e-4, Adapter=5e-3, Baseline=1e-3
- `n_epochs`: Default 5000 (with early stopping)
- `alpha`: Focal loss parameter (default 0.8)

## Batch Experiments

```bash
bash scripts/naroce_pipeline.sh    # Train NAROCE
bash scripts/naroce_nar.sh         # Train NAR only
bash scripts/naroce_adapter.sh     # Train adapters only
bash scripts/baseline.sh           # Train baselines
bash scripts/evaluate_naroce.sh    # Evaluate models
```

## Output

**Models saved to:**
- Baseline: `experiments/baseline/saved_model/{model}/{model}-{dataset}-{seed}.pt`
- NAR: `experiments/naroce/saved_model/nar/{nar_model}-{nar_dataset}-{seed}.pt`
- NAROCE: `experiments/naroce/saved_model/pipeline/{nar_model}-{nar_dataset}-{adapter_model}-{sensor_dataset}-{seed}.pt`

**Logs and results:**
- `experiments/{baseline|naroce}/saved_logs/`

## Troubleshooting

**CUDA out of memory:** Reduce `batch_size` in config or use smaller adapter models

**Cannot load pretrained NAR:** Check `nar_model_seed` in config matches your pretrained NAR

**Dataset not found:** Verify datasets exist in `data/CE_dataset/` (see [`data/README.md`](data/README.md))

## Project Structure

```
naroce/
├── configs/             # JSON configuration files
├── naroce.py            # NAROCE training script
├── baseline.py          # Baseline training script
├── evaluate.py          # Model evaluation
├── data/                # Datasets
├── scripts/             # Batch training scripts
└── experiments/         # Training outputs (models & logs)
```

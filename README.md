# MSL: Mixed Supervised Learning for Fine-Grained Image Classification

This repository contains the official implementation of **"Mix-supervised Learning: A Self-Learning Strategy for Facilitating Data Utilization and Robust Representation"**.

## Overview

MSL (Mixed Supervised Learning) addresses the challenge of fine-grained image classification when labeled training data is limited. Inspired by human learning processes, MSL effectively combines supervised learning on labeled data with pseudo-label-based learning on unlabeled validation data.

### Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.0 (for GPU training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/msl-fine-grained.git
cd msl-fine-grained
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pretrained weights:
```bash
# Download EfficientNet-B3 pretrained on ImageNet
# Place it in the pretrained/ directory as efficientnet-b3.pth
```

## Dataset Preparation

### CUB-200-2011 Dataset

The project uses the CUB-200-2011 dataset for fine-grained bird classification.

1. Download the dataset from [http://www.vision.caltech.edu/datasets/cub_200_2011/](http://www.vision.caltech.edu/datasets/cub_200_2011/)

2. Organize the dataset structure:
```
../datasets/CUB-200(+test)/
├── train/           # Training set (labeled)
│   ├── class001/
│   ├── class002/
│   └── ...
├── val/             # Validation set (unlabeled for semi-supervised learning)
│   ├── class001/
│   ├── class002/
│   └── ...
└── test/            # Test set (for final evaluation)
    ├── class001/
    ├── class002/
    └── ...
```

## Usage

### Training MSL

Train the MSL model on CUB-200 dataset:

```bash
python msl/DRC_MSL.py \
    --dataset CUB200 \
    --batch_size 64 \
    --lr 5e-3 \
    --epochs 30 \
    --msl 1 \
    --confidence 0.9 \
    --lambda_aug 0.5 \
    --epoch_start 4
```

**Key Arguments:**
- `--msl 1`: Enable MSL method (set to 0 for baseline supervised learning)
- `--confidence`: Confidence threshold for pseudo-label filtering (default: 0.9)
- `--lambda_aug`: Weight for feature augmentation loss (default: 0.5)
- `--epoch_start`: Epoch to start using MSL (default: 4)
- `--rewind_threshold`: Accuracy drop threshold for model rewinding (default: 5)

### Training Baseline Methods

#### Semi-Supervised Learning Methods

```bash
# Pseudo-Labeling
python baselines/ssl/pseudo_label.py --dataset CUB200 --batch_size 64 --lr 5e-3

# FixMatch
python baselines/ssl/fixmatch.py --dataset CUB200 --threshold 0.95

# FlexMatch
python baselines/ssl/flexmatch.py --dataset CUB200 --threshold 0.95

# FreeMatch
python baselines/ssl/freematch.py --dataset CUB200

# Mean Teacher
python baselines/ssl/mean_teacher.py --dataset CUB200 --ema_decay 0.999
```

#### Test-Time Adaptation Methods

```bash
# TENT
python baselines/tta/tent.py --dataset CUB200 --tent_lr 1e-3

# CoTTA
python baselines/tta/cotta.py --dataset CUB200 --cotta_lr 1e-3

# SAR
python baselines/tta/sar.py --dataset CUB200 --sar_lr 1e-3 --rho 0.05
```

#### Self-Supervised Learning Methods

```bash
# SimCLR
python baselines/self_supervised/simclr.py --dataset CUB200 --temperature 0.5

# MoCo
python baselines/self_supervised/moco.py --dataset CUB200 --momentum 0.999
```

### Using SLURM

For cluster environments with SLURM:

```bash
sbatch run.slurm
```

Edit `run.slurm` to configure:
- Number of GPUs
- Memory allocation
- Time limit
- Python script and arguments

## Results

### Performance on CUB-200-2011

| Method | Category | Val Acc (%) | Test Acc (%) | Val Top-5 (%) | Test Top-5 (%) |
|--------|----------|-------------|--------------|---------------|----------------|
| **MSL (Ours)** | SSL | **85.14** | **84.48** | **97.19** | **97.09** |
| Pseudo-Labeling | SSL | 82.67 | 83.03 | 96.90 | 96.89 |
| Mean Teacher | SSL | 71.81 | 71.98 | 93.70 | 94.14 |
| FixMatch | SSL | 71.86 | 71.15 | 93.87 | 93.46 |
| FlexMatch | SSL | 72.14 | 70.37 | 94.20 | 94.19 |
| FreeMatch | SSL | 70.74 | 70.52 | 93.92 | 93.41 |
| TENT | TTA | 28.76 | 28.39 | 49.80 | 48.68 |
| SAR | TTA | 29.04 | 27.04 | 51.49 | 49.56 |
| SimCLR | Self-supervised | 56.33 | 60.51 | 84.92 | 87.03 |
| MoCo | Self-supervised | 67.70 | 69.95 | 91.28 | 92.01 |

**All methods use identical training hyperparameters for fair comparison:**
- Batch size: 64
- Learning rate: 5e-3
- Weight decay: 1e-4
- Epochs: 30
- Backbone: EfficientNet-B3

### Why MSL Outperforms Baselines

- **vs. SSL methods**: MSL uses teacher-student consistency without aggressive filtering, better utilizing limited validation data
- **vs. TTA methods**: TTA methods assume pre-trained models and only update BN parameters, insufficient for learning from limited data
- **vs. Self-supervised methods**: Contrastive learning optimizes instance discrimination, not aligned with classification objectives


## References

- **Pseudo-Labeling**: Lee, D. H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. ICML Workshop.
- **FixMatch**: Sohn et al. (2020). FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. NeurIPS.
- **FlexMatch**: Zhang et al. (2021). FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling. NeurIPS.
- **FreeMatch**: Wang et al. (2023). FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning. ICLR.
- **Mean Teacher**: Tarvainen & Valpola (2017). Mean teachers are better role models. NeurIPS.
- **TENT**: Wang et al. (2021). Tent: Fully Test-Time Adaptation by Entropy Minimization. ICLR.
- **SAR**: Niu et al. (2023). Towards Stable Test-Time Adaptation in Dynamic Wild World. ICLR.
- **SimCLR**: Chen et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.
- **MoCo**: He et al. (2020). Momentum Contrast for Unsupervised Visual Representation Learning. CVPR.



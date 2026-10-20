# Dual-Branch Enhancement Network
Polarization Multi-Modal Underwater Image Enhancement Network

## 📖 Introduction
This repository contains the official PyTorch implementation of the dual-branch enhancement network for underwater polarization image enhancement.
The model supports polarization data loading, multi-modal feature fusion, and objective image quality evaluation (PSNR, SSIM).

## 💻 Environment
- OS: Ubuntu 20.04 / Linux
- Python: 3.8.x (Tested on 3.8.10)
- PyTorch: 1.11.0 (Requires CUDA 11.3)
- Torchvision: 0.12.0

> ⚠️ Please strictly follow the version configuration to avoid compatibility errors.

## 📦 Dependencies
| Library | Version | Function |
| ---- | ---- | ---- |
| opencv-python-headless | >=4.13.0 | Basic image reading and matrix transformation |
| scikit-image | ==0.18.3 | Calculate objective evaluation metrics: PSNR, SSIM |
| Pillow | ==9.1.1 | Image preprocessing, cooperate with Torchvision |
| numpy | ==1.22.4 | Numerical calculation and array operation |
| scipy | ==1.10.1 | Scientific calculation and feature processing |
| pytorch-msssim | ==1.0.0 | SSIM loss function based on PyTorch tensor |
| tensorboard | ==2.9.1 | Training loss curve visualization |
| torchsummary / thop | Latest | Count model Params and FLOPs |
| tqdm | ==4.61.2 | Progress bar for training and validation |
| einops | ==0.8.1 | Efficient tensor dimension rearrangement |

### Install
### Create conda environment
```bash
git clone https://github.com/lee-lucia/dual_branch_enhance.git
cd dual_branch_enhance
pip install -r requirements.txt
```
## 🚀 Quick Start

### 1. Dataset Preparation

Organize your polarization underwater dataset, modify the dataset path in the config file.
Dataset structure reference:

```
dataset/
├── train/
│   ├── input/
│   └── target/
└── val/
    ├── input/
    └── target/
```

### 2. Train

```
python train.py
```

- Logs and checkpoints will be saved to the output path set in config.

## ⚠️ Notes

1. CUDA version must match PyTorch 1.11.0, CUDA 11.3 is recommended.
2. Do not upload cache folders (`.ipynb_checkpoints`, `__pycache__`, `.sys`) to git.
3. If you encounter version conflict, rebuild the conda environment.
4. Check GPU memory before training, adjust batch size according to your hardware.

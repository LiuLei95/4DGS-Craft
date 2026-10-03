# 4DGS-Craft: Consistent and Interactive 4D Gaussian Splatting Editing

Official implementation of:

**4DGS-Craft: Consistent and Interactive 4D Gaussian Splatting Editing**

Lei Liu, Can Wang, Zhenghao Chen, Zhenhua Liu, Dong Xu

IEEE Transactions on Circuits and Systems for Video Technology, 2026

Paper:
https://ieeexplore.ieee.org/document/11436015

Project Page:
https://github.com/LiuLei95/4DGS-Craft

---

## Overview

4DGS-Craft is an interactive text-driven editing framework for dynamic scenes represented by 4D Gaussian Splatting.

The framework aims to address three major challenges in 4DGS editing:

- View consistency
- Temporal consistency
- Preservation of non-edited regions

The method contains three main components:

1. **LLM-based Intent Understanding Module**

   Complex user instructions are decomposed into sequential atomic editing operations.

2. **4D-aware InstructPix2Pix**

   A geometry-aware editing module that integrates 4D scene information and multi-view supervision for consistent editing.

3. **Gaussian Selection Mechanism**

   A Gaussian-level selection strategy that tracks editable regions and preserves non-edited areas during optimization.

---

## Method Pipeline

The overall pipeline:

```
User Instruction

        |
        v

LLM Intent Understanding

        |
        v

Atomic Editing Tasks

        |
        v

4D-aware InstructPix2Pix

        |
        v

Gaussian Selector

        |
        v

Edited 4D Gaussian Scene
```

---

## Repository Structure

```
4DGS-Craft/

├── script_final/
│   └── Scripts for final 4DGS-Craft editing experiments
│
├── script_for_3DGS/
│   └── Scripts for 3D Gaussian Splatting based experiments
│
├── src/
│   ├── Deformable-3D-Gaussians/
│   ├── Gaussian editing modules
│   ├── Diffusion based editing modules
│   └── Optimization code
│
└── README.md
```

---

# Installation

## Requirements

Recommended environment:

- Linux
- NVIDIA GPU
- CUDA compatible GPU driver
- Python 3.8+

The experiments were conducted on NVIDIA A800 GPUs.

---

## Create Environment

Example:

```bash
conda create -n 4dgscraft python=3.8
conda activate 4dgscraft
```

Install required packages according to the modules in:

```
src/
```

Please also install the dependencies of the included Gaussian Splatting repositories.

---

# Data Preparation

The experiments use dynamic scene datasets including:

- DyCheck
- N3DV

Prepare the datasets following the original dataset instructions.

Expected structure:

```
data/

├── DyCheck/
│
└── N3DV/
```

---

# Running Experiments

Example scripts are provided in:

```
script_final/
```

Example:

```bash
bash script_final/run_mochi_fox.sh
```

Other provided experiments:

```
run_pillow_AD.sh

run_pillow_shirt.sh

run_sriracha_EM.sh

run_sriracha_green.sh
```

---

# Editing Process

A typical editing pipeline contains:

## 1. User Prompt Understanding

Example:

```
Turn the cat into a fox,
add glasses,
and make its fur white.
```

The LLM module decomposes the instruction:

```
Task 1:
Turn the cat into a fox.

Task 2:
Make the fox white.

Task 3:
Add glasses to the fox.
```

---

## 2. 4D-aware IP2P Optimization

The model generates consistent editing supervision using:

- 4D geometry features
- Multi-view information
- Diffusion-based editing guidance


---

## 3. Gaussian Selection Optimization

The Gaussian Selector identifies editable Gaussian primitives.

Non-edited Gaussians are frozen during optimization to preserve scene details.

---

# Results

The repository reproduces experiments including:

- Object replacement
- Style transfer
- Appearance editing
- Complex instruction following

Example prompts:

```
Turn the cat into a fox.

Make the scene look like a Van Gogh painting.

Change the table color and bookshelf color.
```

---

# Citation

If you find this project useful, please cite:

```bibtex
@ARTICLE{11436015,
author={Liu, Lei and Wang, Can and Chen, Zhenghao and Liu, Zhenhua and Xu, Dong},
journal={IEEE Transactions on Circuits and Systems for Video Technology},
title={4DGS-Craft: Consistent and Interactive 4D Gaussian Splatting Editing},
year={2026}
}
```

---

# Acknowledgement

This project builds upon previous works including:

- 3D Gaussian Splatting
- Deformable Gaussian Splatting
- InstructPix2Pix
- Dynamic NeRF / 4DGS editing methods

---

# License

Please follow the licenses of the corresponding components and datasets.

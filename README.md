# bobble-wobble
A robust computer vision framework optimized to isolate and identify AI-generated synthetic images (AIGC) from authentic photographs under real-world data-degradation conditions.

## The Problem Statement

Traditional machine learning models achieve high accuracy when evaluating clean, high-resolution laboratory data. However, real-world media deployment channels (WhatsApp, TikTok, Facebook) compress, blur, and alter digital signatures. Malicious deepfakes rely on these exact channels to exploit vulnerable demographics by masquerading as urgent public announcements or trusted institutional communications. 

Our pipeline addresses this vulnerability by implementing structural data augmentations directly into the training loop, ensuring our vision backbone identifies core structural forgery cues rather than fragile, high-frequency digital artifacts.

## Repository Architecture

The codebase contains the following functional modules:

*   src/dataset.py: Manages data ingestion pipelines and implements Albumentations pipelines to apply real-world degradation noise (blurring, compression, color-jitter) during training.
*   src/models.py: Structural definition for our binary classification backbone (utilizing a parameterized ConvNeXt-Tiny backbone under the 2B parameter threshold).
*   src/train.py: Handles core training iterations, tracking BCEWithLogitsLoss optimization metrics, and preserving top-performing model weights.
*   src/evaluate_robustness.py: Compiles post-training diagnostic summaries comparing performance variations across clean baseline sets versus transformed test sets.
*   src/utils.py: Analytical helpers calculating accuracy metrics (ROC-AUC, Log-Loss) and exporting false-positive/false-negative error logs.
*   predict.py: Production-level inference gateway designed to parse image directories and generate standardized JSON classification scores.

## Environment & Dependency Setup

1. Clone this repository and ensure Python 3.10+ is initialized.
2. Install framework requirements:
```bash
pip install -r requirements.txt

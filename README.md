# bobble-wobble — Robust AIGC Image Detection

TikTok TechJam submission: detecting AI-generated images with robustness to real-world
post-processing (JPEG re-compression, blur, noise, resizing, color jitter, cropping).

## Project Overview

A binary image classifier (real vs. AI-generated) built on a `timm` vision backbone
(default: ConvNeXt-Tiny, ~28M params — well under the 2B parameter cap), trained with
`BCEWithLogitsLoss`. Robustness is treated as a first-class training and evaluation
concern rather than an afterthought:

- **Training** samples random corruptions (JPEG compression, blur, noise, color jitter,
  cutout) each batch, so the model learns to be invariant to them rather than only
  seeing clean images.
- **Evaluation** runs the *exact* 6 transforms/parameter grid from the hackathon spec
  as separate pipelines and reports a clean-vs-transformed comparison table.

## Repo Structure

```
bobble-wobble/
├── predict.py                    # required inference script (image_dir -> JSON)
├── requirements.txt
└── src/
    ├── dataset.py                # AIGCDataset: reads <root>/real, <root>/fake
    ├── models.py                 # binary classifier builder (timm backbone)
    ├── robustness.py             # the 6 spec transforms + training augmentations
    ├── train.py                  # training loop, BCEWithLogitsLoss, checkpointing
    ├── evaluate_robustness.py    # clean-vs-transformed summary + error analysis
    └── utils.py                  # metrics, JSON export, error-analysis logging
```

## Setup

```bash
pip install -r requirements.txt
```

## Expected Data Layout

```
data/
├── train/
│   ├── real/   *.jpg / *.png ...
│   └── fake/
└── val/
    ├── real/
    └── fake/
```

Any dataset can be reshaped into this layout, e.g. `saberzl/SID_Set`,
`birdy654/cifake-real-and-ai-generated-synthetic-images`, or `WildFake`. The WildFake
subset described in the spec (COCO val2017 + DALL·E Advanced) is a **held-out
demo/benchmark set only** — it is not used for training, per the spec.

## Reproduce Results

**1. Train:**
```bash
python -m src.train --data_dir data --epochs 10 --backbone convnext_tiny --output_dir checkpoints
```
Saves the best checkpoint (by validation ROC-AUC) to `checkpoints/best_model.pth`.

**2. Robustness Evaluation Summary + Error Analysis:**
```bash
python -m src.evaluate_robustness --data_dir data --checkpoint checkpoints/best_model.pth --output_dir reports
```
Writes `reports/robustness_summary.{csv,md}` (clean vs. every transform/severity) and
`reports/error_analysis_clean.json` (top false positives/negatives).

**3. Inference / required deliverable script:**
```bash
python predict.py --image_dir path/to/images --checkpoint checkpoints/best_model.pth --output_json predictions.json
```
Outputs `predictions.json` as a list of `{"image_path": ..., "pred": ...}`, where
`pred` is the probability the image is AI-generated.

## Limitations & Future Work

- Robustness is currently evaluated one transform at a time; real redistribution
  pipelines often **stack** corruptions (e.g. resize + re-compress + crop). A combined
  "worst-case chain" eval would be more representative.
- No adversarial/GAN-aware robustness testing — a detector tuned against one generator
  family may not generalize to newer diffusion models.
- Single backbone/threshold reported here; an ensemble or per-domain calibrated
  threshold would likely reduce false positives on borderline real photos (heavy
  filters, HDR) being flagged as fake.
- Given more time: frequency-domain or noise-residual features (common in forensics
  literature) alongside the RGB backbone, and a calibration pass (temperature scaling)
  so `pred` is a better-calibrated probability, not just a ranking score.

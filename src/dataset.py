# Purpose: Handles reading images from disk and applying the 6 real-world transformations
# specified in the hackathon prompt using Albumentations.
#
# Key Responsibilities:
# 1. Custom PyTorch Dataset class (AIGCDataset) that reads a binary folder layout:
#       <root>/real/*.{jpg,png,...}   -> label 0
#       <root>/fake/*.{jpg,png,...}   -> label 1
# 2. Transform pipelines are imported from src/robustness.py:
#       - get_train_transforms()      heavy random augmentation for training
#       - get_clean_transform()       resize/normalize only, for clean validation
#       - get_robustness_suite()      per-transform/per-severity pipelines, for the
#                                      Robustness Evaluation Summary deliverable

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TO_LABEL = {"real": 0, "fake": 1}  # 0 = authentic, 1 = AI-generated


class AIGCDataset(Dataset):
    """
    Expects a directory structure of:
        root_dir/
            real/
                img001.jpg
                ...
            fake/
                img001.jpg
                ...

    Returns (image_tensor, label_float, image_path) so callers (train/eval scripts)
    can log per-image results for the error-analysis deliverable without a second pass.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for cls_name, label in CLASS_TO_LABEL.items():
            cls_dir = self.root_dir / cls_name
            if not cls_dir.is_dir():
                continue
            for path in sorted(cls_dir.rglob("*")):
                if path.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append((str(path), label))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root_dir}. Expected '{self.root_dir}/real' "
                f"and/or '{self.root_dir}/fake' subfolders."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path)
        if image is None:
            raise IOError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.float32), path

    def set_transform(self, transform):
        """Swap the transform in place — used by evaluate_robustness.py to reuse the
        same underlying file list across every corruption variant without re-scanning disk."""
        self.transform = transform

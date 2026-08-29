# Purpose: Handles reading images from disk and applying the 6 real-world transformations specified in the hackathon prompt using Albumentations.

# Key Responsibilities:
# 1. Custom PyTorch Dataset class (AIGCDataset).
# 2. Separate pipeline functions for Training (heavy random augmentations), Clean Validation, and Transformed Validation (testing specific noise levels like JPEG compression or Gaussian blur).

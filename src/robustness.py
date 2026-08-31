# Purpose: Defines the 6 real-world transformations from the hackathon spec (section 5.2)
# as Albumentations pipelines, so the exact same augmentations can be used both for
# training-time robustness and for the standalone Robustness Evaluation Summary deliverable.
#
# Each transform is exposed at every parameter setting listed in the prompt so we can
# report clean-vs-transformed accuracy per severity level, not just "transform on/off".

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _finish(pipeline, img_size):
    """Append the resize/normalize/tensor steps every eval pipeline needs."""
    return A.Compose(pipeline + [
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_clean_transform(img_size=224):
    """No corruption applied — the 'clean' baseline column in the robustness table."""
    return _finish([], img_size)


def get_robustness_suite(img_size=224):
    """
    Returns an ordered dict: {transform_variant_name: albumentations.Compose}
    covering every parameter setting from the spec table (section 5.2):

        JPEG Compression   quality = 90, 70, 50, 30
        Gaussian Blur      sigma   = 0.5, 1.0, 2.0
        Resize             scale   = 0.5x, 0.25x then upscale back
        Gaussian Noise     sigma   = 0.02, 0.05, 0.10
        Color Jitter       brightness/contrast/saturation +/-20%
        Center Crop        crop 80%

    Used by src/evaluate_robustness.py to build the clean-vs-transformed comparison table.
    """
    suite = {"clean": get_clean_transform(img_size)}

    # JPEG Compression
    for q in (90, 70, 50, 30):
        suite[f"jpeg_q{q}"] = _finish(
            [A.ImageCompression(quality_range=(q, q), p=1.0)], img_size
        )

    # Gaussian Blur (sigma -> approximate odd blur_limit kernel size)
    for sigma in (0.5, 1.0, 2.0):
        k = max(3, int(round(sigma * 6)) | 1)  # odd kernel, roughly 6*sigma
        suite[f"gaussian_blur_sigma{sigma}"] = _finish(
            [A.GaussianBlur(blur_limit=(k, k), sigma_limit=(sigma, sigma), p=1.0)], img_size
        )

    # Resize down then back up (thumbnail generation analog)
    for scale, label in ((0.5, "0.5x"), (0.25, "0.25x")):
        small = max(8, int(img_size * scale))
        suite[f"resize_down_{label}"] = A.Compose([
            A.Resize(small, small),
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    # Gaussian Noise (sigma as fraction of the normalized [0, 1] pixel range)
    for sigma in (0.02, 0.05, 0.10):
        suite[f"gaussian_noise_sigma{sigma}"] = _finish(
            [A.GaussNoise(std_range=(sigma, sigma), p=1.0)], img_size
        )

    # Color Jitter +/-20%
    suite["color_jitter_20pct"] = _finish(
        [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=1.0)], img_size
    )

    # Center Crop 80% (crop to 80% of each dimension, then resize back up to img_size
    # so the model always receives a fixed input shape)
    suite["center_crop_80pct"] = A.Compose([
        A.Resize(img_size, img_size),
        A.CenterCrop(height=int(img_size * 0.8), width=int(img_size * 0.8), p=1.0),
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return suite


def get_train_transforms(img_size=224):
    """
    Heavy random augmentation pipeline for training. Randomly samples from the same
    family of corruptions used in the robustness suite, so the model sees transformed
    inputs during training rather than only at evaluation time.
    """
    hole = int(img_size * 0.2)
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ImageCompression(quality_range=(30, 95), p=1.0),
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.GaussNoise(std_range=(0.05, 0.2), p=1.0),
        ], p=0.6),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
        A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(hole, hole), hole_width_range=(hole, hole), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

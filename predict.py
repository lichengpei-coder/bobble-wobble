"""
AIGC Detection Inference Script

Required deliverable (spec section 5.5.2): takes an image directory as input and
outputs a JSON file of {"image_path": ..., "pred": ...} for each image, where `pred`
is the probability that the image is AI-generated (1.0 = fake, 0.0 = real).

Usage:
    python predict.py --image_dir path/to/images --checkpoint checkpoints/best_model.pth \
                       --output_json predictions.json
"""

import argparse
from pathlib import Path

import cv2
import torch

from src.models import build_model
from src.robustness import get_clean_transform
from src.utils import save_predictions_json

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="AIGC Detection Inference Script")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to directory containing images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained model checkpoint (.pth)")
    parser.add_argument("--output_json", type=str, default="predictions.json", help="Output path for JSON results")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("img_size", 224)


@torch.no_grad()
def predict_directory(model, image_dir, img_size, device, batch_size=32):
    image_paths = sorted(
        p for p in Path(image_dir).rglob("*") if p.suffix.lower() in VALID_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    transform = get_clean_transform(img_size)
    preds = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        tensors = []
        for p in batch_paths:
            image = cv2.imread(str(p))
            if image is None:
                # Skip unreadable files rather than crashing the whole run.
                tensors.append(None)
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensors.append(transform(image=image)["image"])

        valid_idx = [i for i, t in enumerate(tensors) if t is not None]
        if not valid_idx:
            preds.extend([0.5] * len(batch_paths))  # fallback score for unreadable images
            continue

        batch_tensor = torch.stack([tensors[i] for i in valid_idx]).to(device)
        logits = model(batch_tensor).squeeze(1)
        probs = torch.sigmoid(logits).cpu().tolist()

        batch_preds = [0.5] * len(batch_paths)
        for i, prob in zip(valid_idx, probs):
            batch_preds[i] = prob
        preds.extend(batch_preds)

    return image_paths, preds


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, img_size = load_model(args.checkpoint, device)
    image_paths, preds = predict_directory(model, args.image_dir, img_size, device, args.batch_size)
    save_predictions_json(image_paths, preds, args.output_json)

    print(f"Predictions for {len(image_paths)} images saved to {args.output_json}")


if __name__ == "__main__":
    main()

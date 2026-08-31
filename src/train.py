# Purpose: Orchestrates the training loop, validation checks, and checkpoint saving.
#
# Key Responsibilities:
# 1. Sets up PyTorch DataLoaders for training and validation sets.
# 2. Runs Binary Cross-Entropy with Logits loss (BCEWithLogitsLoss).
# 3. Evaluates model performance after every epoch and saves the best model weight file
#    (best_model.pth), selected by validation ROC-AUC.
#
# Expected data layout (see src/dataset.py):
#   <data_dir>/train/real/*, <data_dir>/train/fake/*
#   <data_dir>/val/real/*,   <data_dir>/val/fake/*
#
# Example:
#   python -m src.train --data_dir data --epochs 10 --backbone convnext_tiny

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import AIGCDataset
from src.models import assert_under_param_limit, build_model
from src.robustness import get_clean_transform, get_train_transforms
from src.utils import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train the AIGC binary classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root dir containing train/ and val/ subfolders (each with real/ and fake/)")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet-pretrained weights")
    return parser.parse_args()


def run_epoch(model, loader, device, optimizer=None):
    """Runs one training epoch if `optimizer` is given, otherwise a no-grad eval epoch."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss, n_samples = 0.0, 0
    all_probs, all_labels, all_paths = [], [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels, paths in tqdm(loader, desc="train" if is_train else "val", leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            n_samples += images.size(0)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            all_paths.extend(paths)

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(n_samples, 1)
    return metrics, all_paths, all_labels, all_probs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = AIGCDataset(Path(args.data_dir) / "train", transform=get_train_transforms(args.img_size))
    val_ds = AIGCDataset(Path(args.data_dir) / "val", transform=get_clean_transform(args.img_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.backbone, pretrained=not args.no_pretrained).to(device)
    n_params = assert_under_param_limit(model)
    print(f"Model: {args.backbone} ({n_params:,} params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics, *_ = run_epoch(model, train_loader, device, optimizer)
        val_metrics, *_ = run_epoch(model, val_loader, device, optimizer=None)

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['roc_auc']}"
        )

        val_auc = val_metrics["roc_auc"] or 0.0
        if val_auc >= best_auc:
            best_auc = val_auc
            ckpt_path = output_dir / "best_model.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "img_size": args.img_size,
                "val_auc": best_auc,
                "epoch": epoch,
            }, ckpt_path)
            print(f"  -> saved new best checkpoint to {ckpt_path} (val_auc={best_auc:.4f})")

    print(f"Training complete. Best val ROC-AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()

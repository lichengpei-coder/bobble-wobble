# Purpose: Produces the "Robustness Evaluation Summary" deliverable (spec section 5.5.4) —
# a table comparing model performance on clean validation images versus every transformed
# variant defined in src/robustness.get_robustness_suite().
#
# Also writes the top confidently-wrong false positives/negatives on the clean set,
# covering deliverable 5.5.5 (Error Analysis Note).
#
# Example:
#   python -m src.evaluate_robustness --data_dir data --checkpoint checkpoints/best_model.pth

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import AIGCDataset
from src.models import build_model
from src.robustness import get_robustness_suite
from src.utils import compute_metrics, save_error_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate robustness across all spec transforms")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root dir containing a val/ subfolder (with real/ and fake/)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    all_probs, all_labels, all_paths = [], [], []
    for images, labels, paths in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.tolist())
        all_paths.extend(paths)
    metrics = compute_metrics(all_labels, all_probs)
    return metrics, all_paths, all_labels, all_probs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    img_size = ckpt.get("img_size", 224)
    print(f"Loaded {ckpt['backbone']} checkpoint (val_auc={ckpt.get('val_auc')}, img_size={img_size})")

    val_dir = Path(args.data_dir) / "val"
    dataset = AIGCDataset(val_dir, transform=None)  # transform swapped per-variant below
    suite = get_robustness_suite(img_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for variant_name, transform in suite.items():
        dataset.set_transform(transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
        metrics, paths, labels, probs = evaluate(model, loader, device)
        rows.append({"variant": variant_name, **metrics})
        print(f"{variant_name:>24s}  acc={metrics['accuracy']:.4f}  "
              f"auc={metrics['roc_auc']}  log_loss={metrics['log_loss']}")

        if variant_name == "clean":
            save_error_analysis(paths, labels, probs, output_dir / "error_analysis_clean.json")

    csv_path = output_dir / "robustness_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "accuracy", "roc_auc", "log_loss"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "robustness_summary.md"
    with open(md_path, "w") as f:
        f.write("| Variant | Accuracy | ROC-AUC | Log Loss |\n")
        f.write("|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['variant']} | {r['accuracy']:.4f} | {r['roc_auc']} | {r['log_loss']} |\n")

    print(f"\nSaved: {csv_path}\nSaved: {md_path}\nSaved: {output_dir / 'error_analysis_clean.json'}")


if __name__ == "__main__":
    main()

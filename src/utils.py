# Purpose: Helper utilities for computing metrics, logging results, and formatting outputs.
#
# Key Responsibilities:
# 1. Calculate ROC-AUC, Accuracy, and Log-Loss.
# 2. Export predictions to JSON format (matches the required predict.py output schema:
#    a list of {"image_path": ..., "pred": ...} objects).
# 3. Save error analysis logs (False Positives and False Negatives) for deliverable 5.5.5.

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """
    y_true: iterable of 0/1 ground-truth labels (0 = real, 1 = fake)
    y_prob: iterable of predicted probabilities in [0, 1]

    Returns a dict with roc_auc, accuracy, and log_loss. ROC-AUC needs both classes
    present in y_true; if only one class is present (e.g. a tiny debug batch) it is
    reported as None rather than raising.
    """
    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = None

    try:
        metrics["log_loss"] = log_loss(y_true, y_prob, labels=[0, 1])
    except ValueError:
        metrics["log_loss"] = None

    return metrics


def save_predictions_json(image_paths, preds, output_path):
    """Writes [{"image_path": ..., "pred": ...}, ...] matching the required predict.py schema."""
    results = [
        {"image_path": str(path), "pred": round(float(pred), 4)}
        for path, pred in zip(image_paths, preds)
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def save_error_analysis(image_paths, y_true, y_prob, output_path, threshold: float = 0.5, top_k: int = 50):
    """
    Saves the most confidently-wrong false positives (real predicted as fake) and
    false negatives (fake predicted as real) to a JSON file, sorted by confidence,
    for the Error Analysis Note deliverable (section 5.5.5).
    """
    false_positives, false_negatives = [], []

    for path, label, prob in zip(image_paths, y_true, y_prob):
        pred = 1 if prob >= threshold else 0
        if pred == 1 and label == 0:
            false_positives.append({"image_path": str(path), "pred": round(float(prob), 4)})
        elif pred == 0 and label == 1:
            false_negatives.append({"image_path": str(path), "pred": round(float(prob), 4)})

    # Most confidently wrong first: FPs sorted high->low prob, FNs sorted low->high prob.
    false_positives.sort(key=lambda r: r["pred"], reverse=True)
    false_negatives.sort(key=lambda r: r["pred"])

    report = {
        "num_false_positives": len(false_positives),
        "num_false_negatives": len(false_negatives),
        "false_positives": false_positives[:top_k],
        "false_negatives": false_negatives[:top_k],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    return report

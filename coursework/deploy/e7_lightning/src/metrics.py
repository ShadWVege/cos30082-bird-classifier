"""Evaluation metrics computation derived from a fixed C x C confusion matrix.

Learning Objectives & Authoritative Metric Contract:
- Top-1 Accuracy: Overall sample classification accuracy:
      Top-1 = (sum of correct predictions) / (total test samples N)
- Average Accuracy per Class: Macro-averaged recall across all C classes:
      AvgAcc = (1 / C) * sum_{c=0}^{C-1} (correct_c / support_c)
  CRITICAL DISTINCTION: As specified in the COS30082 assignment brief, this metric is
  unweighted macro recall, NOT one-vs-rest binary accuracy. One-vs-rest binary accuracy
  ((TP + TN) / Total) is trivially 99.5% for an always-negative baseline in a
  balanced 200-class task (where true negatives comprise 199/200 = 99.5% of samples).
  Macro recall weights every bird species equally (1/200th), preventing
  dominant species from masking failure modes on rare species.
- Top-5 Accuracy: Measures whether the true class is among the top 5 largest logits,
  reflecting fine-grained taxonomic grouping (e.g. distinguishing sister warblers).
- Macro-F1: Unweighted arithmetic mean of class-wise F1 scores:
      F1_c = 2 * (Precision_c * Recall_c) / (Precision_c + Recall_c)
"""
import numpy as np


def classification_metrics(targets, logits, num_classes, require_all=True):
    """Compute official classification metrics from integer targets and model logits.

    Args:
        targets (array-like): Ground-truth class indices of shape (N,), values in 0..num_classes-1.
        logits (array-like): Unnormalized model output logits of shape (N, num_classes).
        num_classes (int): Total number of classes (200 for CUB-200).
        require_all (bool): If True, requires every class to have non-zero support (support > 0).
            For official held-out test evaluation, missing-class coverage is treated as an error.

    Returns:
        tuple[dict, list[dict], np.ndarray]:
            - scores: Summary dictionary with 'top1', 'average_accuracy_per_class', 'top5', 'macro_f1'.
            - per_class: List of per-species metrics (support, correct, recall, precision, f1).
            - confusion: Integer confusion matrix of shape (num_classes, num_classes).
    """
    targets, logits = np.asarray(targets, dtype=int), np.asarray(logits)
    if not len(targets) or logits.shape != (len(targets), num_classes):
        raise ValueError("Nonempty targets and N x C logits required")
    if np.any(targets < 0) or np.any(targets >= num_classes) or not np.isfinite(logits).all():
        raise ValueError("Invalid targets or non-finite logits")

    predicted = logits.argmax(axis=1)

    # Efficient vectorized C x C confusion matrix construction using np.add.at
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)

    support = confusion.sum(axis=1)           # Ground-truth count per class (row sums)
    predicted_count = confusion.sum(axis=0)   # Predicted count per class (column sums)

    if require_all and np.any(support == 0):
        raise ValueError("Missing class support: average over all classes would be undefined")

    correct = confusion.diagonal()

    # Per-class recall (accuracy within class) and precision
    recall = np.divide(correct, support, out=np.zeros(num_classes), where=support != 0)
    precision = np.divide(correct, predicted_count, out=np.zeros(num_classes), where=predicted_count != 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(num_classes), where=denom != 0)

    # Top-5 ranking via stable partial sort
    top = np.argsort(-logits, axis=1, kind="stable")[:, :min(5, num_classes)]

    scores = {
        "top1": float(np.mean(targets == predicted)),
        "average_accuracy_per_class": float(recall[support > 0].mean()),
        "top5": float(np.mean(np.any(top == targets[:, None], axis=1))),
        "macro_f1": float(f1.mean()),
        "samples": len(targets),
        "classes_present": int((support > 0).sum()),
    }

    per_class = [
        {
            "label": i,
            "support": int(support[i]),
            "correct": int(correct[i]),
            "accuracy": float(recall[i]) if support[i] else None,
            "precision": float(precision[i]),
            "f1": float(f1[i]),
        }
        for i in range(num_classes)
    ]

    return scores, per_class, confusion


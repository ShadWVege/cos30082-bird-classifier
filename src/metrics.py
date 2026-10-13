"""Required metrics calculated from a fixed C x C confusion matrix."""
import numpy as np


def classification_metrics(targets, logits, num_classes, require_all=True):
    targets, logits = np.asarray(targets, dtype=int), np.asarray(logits)
    if not len(targets) or logits.shape != (len(targets), num_classes):
        raise ValueError("Nonempty targets and N x C logits required")
    if np.any(targets < 0) or np.any(targets >= num_classes) or not np.isfinite(logits).all():
        raise ValueError("Invalid targets or non-finite logits")
    predicted = logits.argmax(axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    support, predicted_count = confusion.sum(axis=1), confusion.sum(axis=0)
    if require_all and np.any(support == 0):
        raise ValueError("Missing class support: average over all classes would be undefined")
    correct = confusion.diagonal()
    recall = np.divide(correct, support, out=np.zeros(num_classes), where=support != 0)
    precision = np.divide(correct, predicted_count, out=np.zeros(num_classes), where=predicted_count != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(num_classes), where=(precision + recall) != 0)
    top = np.argsort(-logits, axis=1, kind="stable")[:, :min(5, num_classes)]
    scores = {"top1": float(np.mean(targets == predicted)),
              "average_accuracy_per_class": float(recall[support > 0].mean()),
              "top5": float(np.mean(np.any(top == targets[:, None], axis=1))),
              "macro_f1": float(f1.mean()), "samples": len(targets),
              "classes_present": int((support > 0).sum())}
    per_class = [{"label": i, "support": int(support[i]), "correct": int(correct[i]),
                  "accuracy": float(recall[i]) if support[i] else None,
                  "precision": float(precision[i]), "f1": float(f1[i])} for i in range(num_classes)]
    return scores, per_class, confusion

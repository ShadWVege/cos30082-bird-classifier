"""Save readable learning curves and error analysis without opening GUI windows."""
from pathlib import Path
import os
from .utils import ROOT
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from .utils import write_csv, write_json


def learning_curves(history, path, smoke=False):
    """Plot train, clean-train and validation loss/Top-1 trajectories.

    The deterministic clean-train curve measures memorization separately
    from the noise introduced by on-the-fly training augmentations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    epochs = [r["epoch"] for r in history]
    for prefix, label in (("train", "Training (augmented if enabled)"), ("clean_train", "Training (clean)"), ("val", "Validation")):
        for axis, metric in zip(axes, ("loss", "top1")):
            axis.plot(epochs, [r[f"{prefix}_{metric}"] for r in history], label=label, marker=".", markersize=5)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Cross-entropy" if metric == "loss" else "Top-1 accuracy")
            axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    if smoke:
        fig.suptitle("Software smoke test only - not assignment performance", fontsize=11)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_analysis(result, names, root, rows, out):
    """Keep machine-readable full confusion and show only legible top errors."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", result["metrics"])
    write_csv(out / "per_class.csv", [{**r, "class_name": names[r["label"]]} for r in result["per_class"]])
    matrix = result["confusion"]
    np.save(out / "confusion.npy", matrix)
    np.savetxt(out / "confusion.csv", matrix, fmt="%d", delimiter=",")
    logits = result["logits"]
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    predicted = logits.argmax(axis=1)
    predictions = [{"name": n, "true_label": y, "true_name": names[y],
                    "predicted_label": int(p), "predicted_name": names[p],
                    "probability": float(probability[i, p]), "correct": bool(y == p)}
                   for i, (n, y, p) in enumerate(zip(result["names"], result["targets"], predicted))]
    write_csv(out / "predictions.csv", predictions)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    pairs = sorted([(int(off_diagonal[i, j]), i, j) for i, j in zip(*np.nonzero(off_diagonal))], reverse=True)[:15]
    write_csv(out / "confused_pairs.csv", [{"true": names[i], "predicted": names[j], "count": count}
                                          for count, i, j in pairs], ["true", "predicted", "count"])
    fig, axis = plt.subplots(figsize=(8, 7), layout="constrained")
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    axis.set(xlabel="Predicted class (zero-based)", ylabel="True class (zero-based)", title="Confusion matrix: counts")
    fig.colorbar(image, ax=axis)
    fig.savefig(out / "confusion.png", dpi=160)
    plt.close(fig)
    if pairs:
        fig, axis = plt.subplots(figsize=(10, 6), layout="constrained")
        axis.barh([f"{names[i]} → {names[j]}" for _, i, j in pairs][::-1], [c for c, _, _ in pairs][::-1])
        axis.set_xlabel("Misclassified images")
        fig.savefig(out / "confused_pairs.png", dpi=160)
        plt.close(fig)
    mistakes = sorted([r for r in predictions if not r["correct"]], key=lambda r: -r["probability"])[:6]
    if mistakes:
        lookup = {r["name"]: r["relative_path"] for r in rows}
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
        for axis in axes.flat:
            axis.axis("off")
        for axis, row in zip(axes.flat, mistakes):
            with Image.open(Path(root) / lookup[row["name"]]) as im:
                axis.imshow(ImageOps.exif_transpose(im).convert("RGB"))
            axis.set_title(f"True: {row['true_name']}\nPredicted: {row['predicted_name']}\nProbability: {row['probability']:.1%}", fontsize=8)
        fig.savefig(out / "errors.png", dpi=160)
        plt.close(fig)

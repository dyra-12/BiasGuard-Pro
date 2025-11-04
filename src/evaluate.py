"""Evaluation utilities.

Loads a trained model (from `models/` by default), evaluates it on the held-out
test split (via `preprocesser.get_dataloaders()`), prints a classification
report and plots the confusion matrix.
"""

from pathlib import Path
import logging
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model_training import evaluate_model
from preprocesser import get_dataloaders

# Logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str]) -> None:
    """Plot a confusion matrix using matplotlib.

    Args:
        cm: square confusion matrix array.
        labels: list of label names (length must match cm.shape[0]).
    """

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.show()


def main(final_model_dir: Path = Path("models")) -> None:
    """Load model and evaluate on the held-out test set.

    The function will attempt to load a model from `final_model_dir`. If the
    directory does not contain a saved model, it will raise an informative
    error.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not final_model_dir.exists() or not any(final_model_dir.iterdir()):
        raise FileNotFoundError(f"Model directory not found or empty: {final_model_dir}")

    logger.info("Loading final model from: %s", final_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(final_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(final_model_dir)
    model.to(device)
    model.eval()

    # Load dataloaders (this will tokenise if necessary and load processed CSVs)
    loaders = get_dataloaders()
    test_loader = loaders["test"]

    # Evaluate using the shared helper from model_training
    metrics, all_preds, all_labels = evaluate_model(model, test_loader, device)

    logger.info("Held-out test metrics: %s", metrics)
    print("\nClassification report:\n")
    print(classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"], digits=4))

    # Confusion matrix and plot
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    logger.info("Confusion matrix:\n%s", cm)
    plot_confusion_matrix(cm, labels=["Class 0", "Class 1"])


if __name__ == "__main__":
    main()


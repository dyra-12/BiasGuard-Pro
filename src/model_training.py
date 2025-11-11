"""Model training utilities.

This module provides a train loop that accepts dataloaders (as produced by
`preprocesser.get_dataloaders`) and a HuggingFace model + tokenizer. It is
designed to be imported and invoked from a training script or notebook; it
does not execute on import.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from preprocesser import compute_metrics_from_preds

# Module logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration container for training runs.

    This dataclass holds commonly-tuned hyperparameters and output
    configuration such as the base model name, number of labels, learning
    rate, regularization, batch size, number of epochs and device.

    Fields are intentionally simple so the object can be serialized to
    JSON for experiment management or used directly when invoking
    ``train_model``.
    """

    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    epochs: int = 3
    warmup_steps: int = 0
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"


def _compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights tensor for CrossEntropyLoss.

    Args:
        labels: 1D numpy array of integer labels.
        device: target torch device.
    """

    vals, counts = np.unique(labels, return_counts=True)
    # Build full list up to max label
    max_label = int(vals.max()) if len(vals) > 0 else 0
    class_counts = [
        int(counts[list(vals).index(i)]) if i in vals else 0
        for i in range(max_label + 1)
    ]
    class_counts = [c if c > 0 else 1 for c in class_counts]
    weights = [sum(class_counts) / c for c in class_counts]
    return torch.tensor(weights, dtype=torch.float, device=device)


def evaluate_model(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[dict, list, list]:
    """Evaluate `model` on `loader` and return metrics and raw preds/labels."""

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics_from_preds(np.array(all_preds), np.array(all_labels))
    return metrics, all_preds, all_labels


def train_model(
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    tokenizer: PreTrainedTokenizerBase,
    config: TrainingConfig,
    checkpoint_dir: Path = Path("models"),
    final_model_dir: Path = Path("models"),
    report_dir: Path = Path("models"),
) -> Dict:
    """Train a sequence classification model using the provided dataloaders.

    Args:
        dataloaders: dict with keys 'train' and 'val' (and optionally 'test').
        tokenizer: tokenizer to save with checkpoints.
        config: TrainingConfig dataclass instance.
        checkpoint_dir/final_model_dir/report_dir: output directories.

    Returns:
        history dict summarizing training.
    """

    device = torch.device(config.device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=config.num_labels
    )
    model.to(device)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Compute class weights from training labels for loss
    all_train_labels = []
    for batch in train_loader:
        all_train_labels.extend(batch["labels"].numpy().tolist())
    class_weights = _compute_class_weights(np.array(all_train_labels), device=device)
    logger.info("Class weights: %s", class_weights.cpu().numpy().tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Estimate total steps
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    history = []

    for epoch in range(1, config.epochs + 1):
        logger.info("Starting epoch %d/%d", epoch, config.epochs)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        num_samples = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            # Move tensors to device (GPU if available). The loader keys
            # are expected to include input_ids, attention_mask and possibly
            # token_type_ids depending on the model/tokenizer.
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits

            # Compute loss with class weights to address class imbalance.
            # The criterion expects raw logits and integer labels.
            loss = criterion(logits, labels)

            # Standard backward/optimizer step
            loss.backward()
            optimizer.step()

            # Step the learning-rate scheduler after optimizer to update
            # the learning rate according to the chosen schedule.
            scheduler.step()

            # Collect running stats for progress display
            preds = torch.argmax(logits, dim=-1)
            running_loss += loss.item() * labels.size(0)
            running_corrects += (preds == labels).sum().item()
            num_samples += labels.size(0)

            batch_loss = running_loss / num_samples
            batch_acc = running_corrects / num_samples
            pbar.set_postfix({"loss": f"{batch_loss:.4f}", "acc": f"{batch_acc:.4f}"})

        epoch_loss = running_loss / (num_samples or 1)
        epoch_acc = running_corrects / (num_samples or 1)

        # Validation
        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        logger.info(
            "Epoch %d train_loss: %.4f | train_acc: %.4f", epoch, epoch_loss, epoch_acc
        )
        logger.info(
            "Epoch %d val_acc: %.4f | val_precision: %.4f | val_recall: %.4f | val_f1: %.4f",
            epoch,
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
        )

        # Save checkpoint
        ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            ckpt_path,
        )
        tokenizer.save_pretrained(
            checkpoint_dir / f"checkpoint_epoch_{epoch}_tokenizer"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            model.save_pretrained(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
            logger.info("-> New best model saved.")

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_acc": epoch_acc,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

    # Save training history
    history_path = (
        report_dir / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(history_path, "w") as hf:
        json.dump(history, hf, indent=2)

    logger.info("Training complete. Best val F1: %s", best_val_f1)
    logger.info("Checkpoints saved to: %s", checkpoint_dir)
    logger.info("Best model saved to: %s", final_model_dir)

    return {
        "history": history,
        "best_val_f1": best_val_f1,
        "checkpoint_dir": checkpoint_dir,
    }


__all__ = ["TrainingConfig", "train_model", "evaluate_model"]

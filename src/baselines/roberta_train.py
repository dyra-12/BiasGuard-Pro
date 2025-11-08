#!/usr/bin/env python3
"""Clean RoBERTa fine-tuning script tailored for this workspace.

Changes from original:
- Reads processed CSVs from `data/processed/biasbios_*.csv`.
- Saves model artifacts under `models/baselines/roberta_baseline`.
- Writes predictions, metrics and a tidy classification-report CSV to `results/`.

This script preserves the robust defaults from the original but removes
Kaggle-specific paths and adds a simple CLI.
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import sys
import time
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Default configuration
MODEL_ID = "roberta-base"
DEFAULT_MODELS_DIR = Path("models/baselines/roberta_baseline")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_PROCESSED_DIR = Path("data/processed")
VEC_SUBDIR = "vectors"

NUM_EPOCHS = 2
LOGGING_STEPS = 10
BATCH_SIZE = 8
GRAD_ACCUM = 2
FP16 = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _detect_column(df: pd.DataFrame, candidates):
    return next((c for c in candidates if c in df.columns), None)


def load_processed_datasets(processed_dir: Path):
    train_path = processed_dir / "biasbios_train.csv"
    val_path = processed_dir / "biasbios_val.csv"
    test_path = processed_dir / "biasbios_test.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Expected processed files in {processed_dir}: biasbios_train/val/test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def prepare_xy(df: pd.DataFrame, role: str = "train"):
    text_candidates = ["text", "cleaned_text", "hard_text", "context", "sentence"]
    label_candidates = ["label", "label_binary", "bias_label", "binary_label"]

    text_col = _detect_column(df, text_candidates)
    label_col = _detect_column(df, label_candidates)

    if text_col is None:
        raise ValueError(f"Could not detect a text column in {role} DataFrame. Columns: {list(df.columns)}")
    if label_col is None:
        logger.warning("No label column found in %s set; classification requires labels.", role)
        raise ValueError(f"Could not detect a label column in {role} DataFrame. Columns: {list(df.columns)}")

    X = df[text_col].astype(str).tolist()
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).tolist()
    return X, y


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


class ConsoleLoggerCallback(TrainerCallback):
    def __init__(self, logfile_path: Path):
        self.logfile_path = logfile_path
        open(self.logfile_path, "a").close()

    def _append(self, s: str):
        with open(self.logfile_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        s = f"[step {state.global_step}] " + json.dumps(logs)
        print(s, flush=True)
        self._append(s)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        s = f"=== Evaluation @ step {state.global_step}, epoch {state.epoch} ==="
        print(s, flush=True)
        self._append(s)
        if metrics:
            for k, v in metrics.items():
                line = f"  {k}: {v}"
                print(line, flush=True)
                self._append(line)


def save_classification_report_csv(report_dict, out_path: Path):
    rows = []
    for key, vals in report_dict.items():
        if isinstance(vals, dict):
            row = {"label": key}
            row.update(vals)
            rows.append(row)
        else:
            rows.append({"label": key, "value": vals})
    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_confusion_matrix(y_true, y_pred, out_csv: Path, out_png: Path, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels if labels else None, columns=labels if labels else None)
    cm_df.to_csv(out_csv, index=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return cm, cm_df


def extract_and_save_vectors(model, dataset, out_dir: Path, split_name: str, device, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_vecs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.get("label", None)
            if isinstance(labels, torch.Tensor):
                all_labels.append(labels.cpu().numpy())

            inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            cls_vecs = outputs.hidden_states[-1][:, 0, :]
            all_vecs.append(cls_vecs.float().cpu().numpy())

    vecs = np.concatenate(all_vecs, axis=0) if all_vecs else np.empty((0, model.config.hidden_size), dtype=np.float32)
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{split_name}_vectors.npy", vecs)
    np.save(out_dir / f"{split_name}_labels.npy", labels_np)
    return vecs, labels_np


def main(
    model_id: str = MODEL_ID,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    grad_accum: int = GRAD_ACCUM,
    fp16: bool = FP16,
):
    start_time = time.time()

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    vec_dir = models_dir / VEC_SUBDIR
    vec_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_processed_datasets(processed_dir)
    X_train, y_train = prepare_xy(train_df, role="train")
    X_val, y_val = prepare_xy(val_df, role="val")
    X_test, y_test = prepare_xy(test_df, role="test")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    # Build datasets and tokenize
    train_dataset = Dataset.from_dict({"text": X_train, "label": y_train})
    val_dataset = Dataset.from_dict({"text": X_val, "label": y_val})
    test_dataset = Dataset.from_dict({"text": X_test, "label": y_test})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(models_dir),
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=str(results_dir),
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        seed=42,
        report_to=None,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        fp16=fp16,
    )

    log_path = models_dir / "run_log.txt"
    callback = ConsoleLoggerCallback(log_path)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[callback],
    )

    logger.info("Starting training")
    trainer.train()

    # Save model & tokenizer
    trainer.save_model(str(models_dir))
    tokenizer.save_pretrained(str(models_dir))

    # (Evaluation and artifact extras removed; evaluation handled centrally.)

    elapsed = time.time() - start_time
    logger.info("Run completed in %.1f seconds", elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on BiasBios processed data and produce reports")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--processed_dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--models_dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    main(
        model_id=args.model_id,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        fp16=args.fp16,
    )
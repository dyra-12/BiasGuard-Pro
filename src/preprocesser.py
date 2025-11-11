"""Preprocessing utilities: tokenize datasets and build PyTorch dataloaders.

This module reads the preprocessed CSVs produced by `data_loader.py` (in
`data/processed`), tokenizes text using a HuggingFace tokenizer, and returns
PyTorch DataLoaders for training, validation and held-out test.

It provides a single convenient function `get_dataloaders(...)` that can be
used in training scripts or notebooks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix  # type: ignore
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Module logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Default configuration
SEED = 42
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
MAX_LENGTH = 256
BATCH_SIZE = 16


class TextDataset(Dataset):
    """Simple torch Dataset for tokenized inputs.

    Expects encodings (the mapping returned by a HF tokenizer) and labels.
    """

    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def compute_metrics_from_preds(preds, labels):
    """Compute basic binary classification metrics from predictions.

    Args:
        preds: Iterable of predicted class labels (0/1).
        labels: Iterable of true class labels (0/1).

    Returns:
        Dict with keys: accuracy, precision, recall, f1.
    """

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_json(obj, path: Path) -> None:
    """Persist `obj` as JSON to `path`.

    Args:
        obj: JSON-serializable Python object.
        path: Destination file path.
    """

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _detect_column(df: pd.DataFrame, candidates):
    """Return the first column name in `candidates` that exists in `df`.

    Args:
        df: DataFrame to inspect.
        candidates: Iterable of candidate column names in preference order.

    Returns:
        Matching column name or None if none found.
    """

    return next((c for c in candidates if c in df.columns), None)


def _load_processed_dfs(
    processed_dir: Path = Path("data/processed"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the processed train/val/test CSVs written by `data_loader.py`.

    Expects files: biasbios_train.csv, biasbios_val.csv,
    biasbios_test.csv. Raises FileNotFoundError if missing.
    """

    train_path = processed_dir / "biasbios_train.csv"
    val_path = processed_dir / "biasbios_val.csv"
    test_path = processed_dir / "biasbios_test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run data_loader first.")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing {val_path}. Run data_loader first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}. Run data_loader first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def tokenize_texts(
    tokenizer: AutoTokenizer, texts, max_length: int = MAX_LENGTH
) -> dict:
    """Tokenize a list of texts using the provided HuggingFace tokenizer.

    Args:
        tokenizer: HF tokenizer instance.
        texts: Iterable of text strings.
        max_length: Maximum token sequence length.

    Returns:
        Tokenizer encoding dict (input_ids, attention_mask, etc.).
    """

    return tokenizer(
        list(texts), truncation=True, padding="max_length", max_length=max_length
    )


def get_dataloaders(
    tokenizer_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    processed_dir: Path = Path("data/processed"),
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    seed: int = SEED,
) -> Dict[str, DataLoader]:
    """Return a dict with PyTorch DataLoaders: {'train','val','test'}.

    If DataFrames are not provided, the function will load the processed CSVs
    from `processed_dir`.

    The returned DataLoaders yield batches with tensors ready for a HF
    Transformers Trainer-style model (input_ids, attention_mask, labels).
    """

    logger.info("Loading or receiving DataFrames for dataloader creation")

    if train_df is None or val_df is None or test_df is None:
        tdf, vdf, tef = _load_processed_dfs(processed_dir=processed_dir)
        train_df = train_df or tdf
        val_df = val_df or vdf
        test_df = test_df or tef

    # Detect text/label columns
    text_candidates = ["text", "cleaned_text", "hard_text", "context", "sentence"]
    label_candidates = ["label", "label_binary", "bias_label", "binary_label"]

    train_text_col = _detect_column(train_df, text_candidates) or "text"
    val_text_col = _detect_column(val_df, text_candidates) or "text"

    # For test (held-out), original BiasBios columns may be preserved; detect
    test_text_col = _detect_column(test_df, text_candidates) or "text"
    test_label_col = _detect_column(test_df, label_candidates) or "label"

    logger.info(
        "Using text columns: train=%s val=%s test=%s",
        train_text_col,
        val_text_col,
        test_text_col,
    )
    logger.info("Using test label column: %s", test_label_col)

    # Ensure labels exist and are ints for train/val
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    # For test, rename detected label/text cols to canonical names to simplify downstream
    if test_text_col != "text":
        # Normalize held-out test column names to the canonical 'text' to
        # avoid branching logic downstream when loading the test split.
        test_df = test_df.rename(columns={test_text_col: "text"})
    if test_label_col != "label":
        # Likewise ensure the held-out label column is simply 'label'. This
        # keeps the loader code consistent even if BiasBios preserved
        # additional legacy column names.
        test_df = test_df.rename(columns={test_label_col: "label"})

    # Tokenizer
    logger.info("Loading tokenizer: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize
    logger.info("Tokenizing datasets (this may take a while)...")
    train_enc = tokenize_texts(
        tokenizer, train_df["text"].astype(str).tolist(), max_length=max_length
    )
    val_enc = tokenize_texts(
        tokenizer, val_df["text"].astype(str).tolist(), max_length=max_length
    )
    test_enc = tokenize_texts(
        tokenizer, test_df["text"].astype(str).tolist(), max_length=max_length
    )

    # Build datasets and loaders
    train_dataset = TextDataset(train_enc, train_df["label"].tolist())
    val_dataset = TextDataset(val_enc, val_df["label"].tolist())
    test_dataset = TextDataset(test_enc, test_df["label"].astype(int).tolist())

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(
        "DataLoaders ready. Batches: train=%d val=%d test=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


if __name__ == "__main__":
    # Quick smoke test: build dataloaders and report one batch shapes
    loaders = get_dataloaders()
    batch = next(iter(loaders["train"]))
    logger.info("Example batch keys: %s", list(batch.keys()))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            logger.info("%s shape: %s", k, tuple(v.shape))
        else:
            logger.info("%s type: %s", k, type(v))

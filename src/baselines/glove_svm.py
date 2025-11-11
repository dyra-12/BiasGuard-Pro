#!/usr/bin/env python
"""Embedding-based baseline: GloVe/Word2Vec + LinearSVC.

This cleaned version uses processed CSVs located in `data/processed`:

- data/processed/biasbios_train.csv
- data/processed/biasbios_val.csv
- data/processed/biasbios_test.csv

It trains a LinearSVC on mean-pooled word embeddings and persists:

- models/baselines/glove_svm_model.joblib
- models/baselines/glove_svm_metadata.json
- results/glove_svm_predictions.csv
- results/glove_svm_metrics.json
- results/glove_svm_classification_report.csv

If embedding loading fails, the script exits gracefully.
"""

import argparse
import json
import logging
from pathlib import Path

import gensim.downloader as api
import joblib
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

# Configuration
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_MODELS_DIR = Path("models/baselines")
DEFAULT_RESULTS_DIR = Path("results")
SEED = 42

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _detect_column(df: pd.DataFrame, candidates):
    """Return the first candidate column name present in `df`.

    This helper centralizes the common pattern of trying multiple possible
    column names for text/label fields so callers can remain simple.

    Args:
        df: DataFrame to inspect.
        candidates: Iterable of candidate column names in preference order.

    Returns:
        The first matching column name or ``None`` if none found.
    """

    return next((c for c in candidates if c in df.columns), None)


def load_processed_datasets(processed_dir: Path):
    train_path = processed_dir / "biasbios_train.csv"
    val_path = processed_dir / "biasbios_val.csv"
    test_path = processed_dir / "biasbios_test.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected processed files in {processed_dir}: biasbios_train/val/test.csv"
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def prepare_xy(df: pd.DataFrame, role: str = "train"):
    """Detect text and label columns and return (X:list[str], y:list[int]).

    Args:
        df: DataFrame containing text and label fields.
        role: Human-readable role string used in error messages (train/val/test).

    Returns:
        Tuple of (X, y) where X is a list of strings and y a list of ints.
    """

    text_candidates = ["text", "cleaned_text", "hard_text", "context", "sentence"]
    label_candidates = ["label", "label_binary", "bias_label", "binary_label"]

    text_col = _detect_column(df, text_candidates)
    label_col = _detect_column(df, label_candidates)

    if text_col is None:
        raise ValueError(
            f"Could not detect a text column in {role} DataFrame. Columns: {list(df.columns)}"
        )
    if label_col is None:
        logger.warning(
            "No label column found in %s set; classification/training requires labels.",
            role,
        )
        raise ValueError(
            f"Could not detect a label column in {role} DataFrame. Columns: {list(df.columns)}"
        )

    X = df[text_col].astype(str).tolist()
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).tolist()
    return X, y


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def save_predictions_and_metrics(
    model_name: str, y_true, y_pred, texts, results_dir: Path
):
    results_dir.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "text": texts})
    preds_out = results_dir / f"{model_name}_predictions.csv"
    preds_df.to_csv(preds_out, index=False)

    metrics = compute_metrics(y_true, y_pred)
    metrics_out = results_dir / f"{model_name}_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(y_true, y_pred, output_dict=True)
    report_out = results_dir / f"{model_name}_classification_report.csv"
    # convert report dict to tidy CSV
    rows = []
    for k, v in report.items():
        if isinstance(v, dict):
            row = {"label": k}
            row.update(v)
            rows.append(row)
        else:
            rows.append({"label": k, "value": v})
    pd.DataFrame(rows).to_csv(report_out, index=False)

    logger.info("Saved predictions -> %s", preds_out)
    logger.info("Saved metrics -> %s", metrics_out)
    logger.info("Saved classification report CSV -> %s", report_out)


def load_embedding_model():
    """Try to load a suitable gensim embedding model and return (model, name)."""
    try:
        logger.info("Loading GloVe embeddings (glove-wiki-gigaword-300)...")
        m = api.load("glove-wiki-gigaword-300")
        return m, "glove-wiki-gigaword-300"
    except Exception as e:
        logger.warning("GloVe 300 load failed: %s", e)
    try:
        logger.info("Loading Word2Vec (word2vec-google-news-300)...")
        m = api.load("word2vec-google-news-300")
        return m, "word2vec-google-news-300"
    except Exception as e:
        logger.warning("Word2Vec load failed: %s", e)
    try:
        logger.info("Loading GloVe 100D (glove-wiki-gigaword-100)...")
        m = api.load("glove-wiki-gigaword-100")
        return m, "glove-wiki-gigaword-100"
    except Exception as e:
        logger.warning("Small GloVe load failed: %s", e)
    return None, None


def document_to_vector(text: str, embedding_model, embedding_dim: int):
    """Convert a document to a single mean-pooled embedding vector.

    Words not present in the embedding vocabulary are skipped. When no
    known words are present the function returns a zero-vector to avoid
    failing downstream classifiers.

    Args:
        text: Input string.
        embedding_model: Gensim KeyedVectors-like object.
        embedding_dim: Expected embedding dimensionality.

    Returns:
        1D NumPy array of length embedding_dim.
    """

    if not text:
        return np.zeros(embedding_dim, dtype=np.float32)
    words = text.split()
    key_to_index = getattr(embedding_model, "key_to_index", {})
    vecs = []
    for w in words:
        if w in key_to_index:
            vecs.append(embedding_model[w])
    if len(vecs) > 0:
        return np.mean(vecs, axis=0)
    return np.zeros(embedding_dim, dtype=np.float32)


def texts_to_vectors(texts, embedding_model, embedding_dim=300):
    """Batch-convert a list of texts to an (n_texts, embedding_dim) array.

    This wrapper pre-allocates the array for efficiency and calls
    ``document_to_vector`` for each row.
    """

    vectors = np.zeros((len(texts), embedding_dim), dtype=np.float32)
    for i, t in enumerate(texts):
        vectors[i] = document_to_vector(t, embedding_model, embedding_dim)
    return vectors


def main(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    seed: int = SEED,
):
    np.random.seed(seed)

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed datasets from %s", processed_dir)
    train_df, val_df, test_df = load_processed_datasets(processed_dir)

    X_train, y_train = prepare_xy(train_df, role="train")
    X_val, y_val = prepare_xy(val_df, role="val")
    X_test, y_test = prepare_xy(test_df, role="test")

    model_name = "glove_svm"

    embedding_model, embedding_name = load_embedding_model()
    if embedding_model is None:
        logger.error("No embedding model available; aborting embedding baseline.")
        return

    embedding_dim = int(getattr(embedding_model, "vector_size", 300))
    logger.info("Using embedding: %s (dim=%d)", embedding_name, embedding_dim)

    logger.info("Converting texts to embedding vectors...")
    X_train_emb = texts_to_vectors(X_train, embedding_model, embedding_dim)
    X_val_emb = texts_to_vectors(X_val, embedding_model, embedding_dim)
    X_test_emb = texts_to_vectors(X_test, embedding_model, embedding_dim)

    logger.info("Training LinearSVC on embeddings...")
    clf = LinearSVC(random_state=seed, class_weight="balanced", max_iter=5000)
    clf.fit(X_train_emb, y_train)

    # Save model and metadata
    svm_out = models_dir / f"{model_name}_model.joblib"
    joblib.dump(clf, svm_out, compress=3)

    meta = {
        "pipeline": model_name,
        "embedding_name": embedding_name,
        "embedding_dim": embedding_dim,
        "model_path": str(svm_out),
    }
    meta_out = models_dir / f"{model_name}_metadata.json"
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved model -> %s", svm_out)
    logger.info("Saved metadata -> %s", meta_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GloVe/Word2Vec + LinearSVC baseline on BiasBios processed data"
    )
    parser.add_argument("--processed_dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--models_dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    main(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        seed=args.seed,
    )

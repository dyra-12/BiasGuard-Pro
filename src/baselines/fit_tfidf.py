"""TF-IDF + Logistic Regression baseline training script.

This script expects the processed CSVs in `data/processed` with the
following filenames:

- data/processed/biasbios_train.csv
- data/processed/biasbios_val.csv
- data/processed/biasbios_test.csv

It trains a TF-IDF vectorizer + Logistic Regression classifier and
persists artifacts and evaluation outputs into the repository:

- models/baselines/<model>_vectorizer.joblib
- models/baselines/<model>_model.joblib
- results/<model>_predictions.csv (per-example predictions)
- results/<model>_metrics.json (overall metrics)
- results/<model>_classification_report.csv (human readable CSV)

The script is written to be robust to minor column-name differences in
the processed CSVs (it will try to autodetect label/text columns).
"""

from pathlib import Path
import json
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_MODELS_DIR = Path("models/baselines")
DEFAULT_RESULTS_DIR = Path("results")
SEED = 42

# Module logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _detect_column(df: pd.DataFrame, candidates):
    """Return first matching column name from candidates present in `df`.

    If none match, returns None.
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

    role is used only for nicer error messages.
    """
    text_candidates = ["text", "cleaned_text", "hard_text", "context", "sentence"]
    label_candidates = ["label", "label_binary", "bias_label", "binary_label"]

    text_col = _detect_column(df, text_candidates)
    label_col = _detect_column(df, label_candidates)

    if text_col is None:
        raise ValueError(f"Could not detect a text column in {role} DataFrame. Columns: {list(df.columns)}")
    if label_col is None:
        # It's possible the test set keeps a different label name; if fully missing, raise
        logger.warning("No label column found in %s set; classification/training requires labels.", role)
        raise ValueError(f"Could not detect a label column in {role} DataFrame. Columns: {list(df.columns)}")

    X = df[text_col].astype(str).tolist()
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).tolist()

    return X, y


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_classification_report_csv(report_dict, out_path: Path):
    """Convert sklearn classification_report (output_dict=True) to a tidy CSV."""
    # report_dict is mapping label -> metrics, plus 'accuracy','macro avg','weighted avg'
    rows = []
    for key, vals in report_dict.items():
        if isinstance(vals, dict):
            row = {"label": key}
            row.update(vals)
            rows.append(row)
        else:
            # scalar like accuracy
            rows.append({"label": key, "value": vals})

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def main(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    seed: int = SEED,
):
    np.random.seed(seed)

    # Create dirs
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed datasets from %s", processed_dir)
    train_df, val_df, test_df = load_processed_datasets(processed_dir)

    logger.info("Preparing train/val/test splits")
    X_train, y_train = prepare_xy(train_df, role="train")
    X_val, y_val = prepare_xy(val_df, role="val")
    X_test, y_test = prepare_xy(test_df, role="test")

    # Vectorizer + model
    model_name = "tfidf_logreg"

    logger.info("Fitting TF-IDF vectorizer")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english", min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    logger.info("Training Logistic Regression")
    clf = LogisticRegression(random_state=seed, max_iter=1000, class_weight="balanced")
    clf.fit(X_train_tfidf, y_train)

    # Evaluate on test set
    logger.info("Predicting on test set")
    y_test_pred = clf.predict(X_test_tfidf)

    # Save per-example predictions
    preds_df = pd.DataFrame({"y_true": y_test, "y_pred": y_test_pred, "text": X_test})
    preds_out = results_dir / f"{model_name}_predictions.csv"
    preds_df.to_csv(preds_out, index=False)
    logger.info("Saved predictions -> %s", preds_out)

    # Compute and save metrics + classification report
    metrics = compute_metrics(y_test, y_test_pred)
    metrics_out = results_dir / f"{model_name}_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics -> %s", metrics_out)

    report = classification_report(y_test, y_test_pred, output_dict=True)
    report_csv_out = results_dir / f"{model_name}_classification_report.csv"
    save_classification_report_csv(report, report_csv_out)
    logger.info("Saved classification report CSV -> %s", report_csv_out)

    # Persist vectorizer and model
    vec_out = models_dir / f"{model_name}_vectorizer.joblib"
    mdl_out = models_dir / f"{model_name}_model.joblib"
    joblib.dump(tfidf, vec_out, compress=3)
    joblib.dump(clf, mdl_out, compress=3)
    logger.info("Saved vectorizer -> %s", vec_out)
    logger.info("Saved model      -> %s", mdl_out)

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TF-IDF + LogisticRegression baseline on BiasBios processed data")
    parser.add_argument("--processed_dir", type=Path, default=DEFAULT_PROCESSED_DIR, help="Path to processed CSVs")
    parser.add_argument("--models_dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory to save models")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory to save results")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    main(processed_dir=args.processed_dir, models_dir=args.models_dir, results_dir=args.results_dir, seed=args.seed)
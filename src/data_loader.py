"""Data loader utilities for BiasBios + synthetic dataset.

This module loads the raw CSVs from `data/raw`, detects the appropriate
text/label columns, performs a held-out split on the BiasBios dataset, and
creates a combined train/validation pool with the synthetic data. The held-
out BiasBios test split is saved to `data/processed/biasbios_test.csv`.

The code is organized as small functions with clear docstrings so it is
easy to reuse and read in a research context.
"""

import logging
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional in some envs
    torch = None
    TORCH_AVAILABLE = False

from sklearn.model_selection import train_test_split

# -------------------------
# Configuration / constants
# -------------------------
SEED: int = globals().get("SEED", 42)
INPUT_DIR: Path = Path("data/raw")
BIASBIOS_CSV: Path = INPUT_DIR / "master_dataset.csv"
SYNTHETIC_CSV: Path = INPUT_DIR / "synthetic_career_data.csv"

# Splits
BIASBIOS_TEST_FRAC: float = 0.10
VAL_FRAC: float = 0.10

# Configure module logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    """Set seeds for reproducibility.

    Makes best-effort to seed numpy and torch (if available). Does not raise
    if torch is missing or fails to initialize.
    """

    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            # If torch fails (binary incompatibilities), continue gracefully
            pass


def _detect_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return first matching column name from candidates present in `df`.

    Args:
        df: DataFrame in which to search.
        candidates: Ordered candidates (preference order).

    Returns:
        Column name or None if not found.
    """

    return next((c for c in candidates if c in df.columns), None)


def load_and_prepare(
    bias_csv: Path = BIASBIOS_CSV,
    synth_csv: Path = SYNTHETIC_CSV,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Load CSVs, detect columns, perform splits, and return datasets.

    Returns a tuple: (train_df, val_df, biasbios_test_df, text_col, label_col)
    """

    _seed_everything(seed)

    if not bias_csv.exists():
        raise FileNotFoundError(f"BiasBios CSV not found at {bias_csv}")
    if not synth_csv.exists():
        raise FileNotFoundError(f"Synthetic CSV not found at {synth_csv}")

    biasbios_df = pd.read_csv(bias_csv)
    synthetic_df = pd.read_csv(synth_csv)

    # Detect label and text columns using sensible preference orders
    LABEL_COL_CANDIDATES: List[str] = [
        "label_binary",
        "bias_label",
        "binary_label",
        "label",
    ]
    TEXT_COL_CANDIDATES: List[str] = [
        "text",
        "cleaned_text",
        "hard_text",
        "context",
        "sentence",
    ]

    bias_label_col = _detect_column(biasbios_df, LABEL_COL_CANDIDATES)
    synth_label_col = _detect_column(synthetic_df, LABEL_COL_CANDIDATES)
    if bias_label_col is None or synth_label_col is None:
        raise ValueError("Could not find label columns in CSVs. Inspect CSV columns")

    bias_text_col = _detect_column(biasbios_df, TEXT_COL_CANDIDATES)
    synth_text_col = _detect_column(synthetic_df, TEXT_COL_CANDIDATES)
    if bias_text_col is None or synth_text_col is None:
        raise ValueError(
            "Could not find text column in one of the CSVs. Inspect CSV columns"
        )

    # Ensure labels are numeric and convert to ints. If any missing/NA values are
    # present, warn and fill with 0 to avoid crashing; this keeps pipeline robust
    # but may mask dataset issues — inspect logs if this happens.
    biasbios_df[bias_label_col] = pd.to_numeric(
        biasbios_df[bias_label_col], errors="coerce"
    )
    if biasbios_df[bias_label_col].isna().any():
        logger.warning(
            "Found %d missing labels in BiasBios; filling with 0",
            int(biasbios_df[bias_label_col].isna().sum()),
        )
    biasbios_df[bias_label_col] = biasbios_df[bias_label_col].fillna(0).astype(int)

    synthetic_df[synth_label_col] = pd.to_numeric(
        synthetic_df[synth_label_col], errors="coerce"
    )
    if synthetic_df[synth_label_col].isna().any():
        logger.warning(
            "Found %d missing labels in synthetic data; filling with 0",
            int(synthetic_df[synth_label_col].isna().sum()),
        )
    synthetic_df[synth_label_col] = synthetic_df[synth_label_col].fillna(0).astype(int)

    # Train/held-out split (held-out is from BiasBios only)
    stratify_col = (
        biasbios_df[bias_label_col]
        if biasbios_df[bias_label_col].nunique() > 1
        else None
    )

    train_val_df, biasbios_test_df = train_test_split(
        biasbios_df,
        test_size=BIASBIOS_TEST_FRAC,
        random_state=seed,
        stratify=stratify_col,
    )

    # Create combined train+val pool with synthetic data
    # Preserve BiasBios metadata columns for BiasBios-origin rows while
    # keeping a canonical 'text' and 'label' column used downstream.
    desired_biasbios_cols = [
        "hard_text",
        "profession",
        "gender",
        "profession_name",
        "text",
        "bias_score",
        "label_binary",
        "label_multiclass",
    ]

    # Ensure biasbios part has the desired columns (add missing columns as NA)
    bias_part = train_val_df.copy()
    for c in desired_biasbios_cols:
        if c not in bias_part.columns:
            bias_part[c] = pd.NA

    # Create a canonical 'label' column for bias rows from the BiasBios
    # dataset. Use 'label_binary' as the canonical label when present; this
    # ensures training uses the binary gendered label consistently.
    if "label_binary" in bias_part.columns:
        bias_part["label"] = (
            pd.to_numeric(bias_part["label_binary"], errors="coerce")
            .fillna(0)
            .astype(float)
        )
    else:
        bias_part["label"] = (
            pd.to_numeric(bias_part[bias_label_col], errors="coerce")
            .fillna(0)
            .astype(float)
        )

    # For synthetic rows, build a DataFrame with the same biasbios columns filled with NA
    synthetic_part = pd.DataFrame(columns=desired_biasbios_cols)
    synthetic_part = synthetic_part.reindex(range(len(synthetic_df))).copy()
    for c in desired_biasbios_cols:
        synthetic_part[c] = pd.NA
    # add canonical text/label from synthetic
    synthetic_part["text"] = synthetic_df[synth_text_col].astype(str).values
    synthetic_part["label"] = synthetic_df[synth_label_col].astype(float).values

    # Add source markers
    bias_part["source"] = "biasbios"
    synthetic_part["source"] = "synthetic"

    combined_pool = pd.concat(
        [bias_part[desired_biasbios_cols + ["label", "source"]], synthetic_part],
        ignore_index=True,
    )
    combined_pool = combined_pool.sample(frac=1, random_state=seed).reset_index(
        drop=True
    )
    # Fill missing BiasBios-specific binary label for synthetic rows by
    # copying the canonical 'label' into 'label_binary' where absent. This
    # ensures the 'label_binary' column is populated for all training rows
    # (useful for inspection and compatibility with older code expecting it).
    if "label_binary" in combined_pool.columns:
        combined_pool["label_binary"] = pd.to_numeric(
            combined_pool["label_binary"], errors="coerce"
        )
        combined_pool["label_binary"] = (
            combined_pool["label_binary"].fillna(combined_pool["label"]).astype(int)
        )
    else:
        combined_pool["label_binary"] = combined_pool["label"].astype(int)

    # At this point combined_pool contains both BiasBios and synthetic rows
    # with a unified schema. The 'source' column lets downstream tools split
    # or filter on origin when needed (e.g., computing domain-specific stats).

    stratify_pool = (
        combined_pool["label"] if combined_pool["label"].nunique() > 1 else None
    )

    train_df, val_df = train_test_split(
        combined_pool, test_size=VAL_FRAC, random_state=seed, stratify=stratify_pool
    )

    # Drop rows with missing critical fields (text or label) and report by source
    def _clean(df_in: pd.DataFrame, name: str) -> pd.DataFrame:
        """Remove rows with missing critical fields and normalize types.

        This inner helper is small and used only during preparation. It logs
        a summary of dropped rows grouped by source to aid debugging of
        data quality issues.
        """

        dfc = df_in.copy()
        # Ensure types
        dfc["label"] = pd.to_numeric(dfc["label"], errors="coerce")
        dfc["text"] = dfc["text"].astype(object)
        missing_mask = dfc["text"].isna() | dfc["label"].isna()
        if missing_mask.any():
            by_source = dfc.loc[missing_mask, "source"].value_counts(dropna=False)
            logger.warning(
                "Dropping %d %s rows with missing text/label. By source: %s",
                int(missing_mask.sum()),
                name,
                by_source.to_dict(),
            )
            dfc = dfc.loc[~missing_mask].reset_index(drop=True)
        # Keep label as int
        dfc["label"] = dfc["label"].astype(int)
        return dfc

    train_df = _clean(train_df, "train")
    val_df = _clean(val_df, "val")

    return train_df, val_df, biasbios_test_df, bias_text_col, bias_label_col


def save_heldout(
    biasbios_test_df: pd.DataFrame, output_dir: Path = Path("data/processed")
) -> Path:
    """Save the held-out BiasBios test split and return the path.

    The function preserves original BiasBios columns.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "biasbios_test.csv"
    biasbios_test_df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    """Run end-to-end data preparation and persist the held-out test split.

    Logs brief summaries for quick verification.
    """

    logger.info("Loading CSVs from: %s", INPUT_DIR)

    train_df, val_df, biasbios_test_df, text_col, label_col = load_and_prepare()

    logger.info(
        "BiasBios rows: %d",
        len(biasbios_test_df) + len(train_df) + len(val_df) - len(val_df),
    )
    logger.info("Using BiasBios label column: %s", label_col)
    logger.info("Using BiasBios text column: %s", text_col)

    # Quick distributions for researcher glance
    logger.info(
        "Label distribution (train):\n%s", train_df["label"].value_counts().to_string()
    )

    logger.info(
        "Label distribution (val):\n%s", val_df["label"].value_counts().to_string()
    )

    logger.info(
        "Held-out BiasBios test distribution:\n%s",
        biasbios_test_df[label_col].value_counts().to_string(),
    )

    out_path = save_heldout(biasbios_test_df)
    logger.info("Saved held-out BiasBios test to: %s", out_path)

    # Also persist preprocessed training and validation sets with minimal schema
    # (text, label) to avoid expected NaNs in BiasBios-only metadata columns
    # when mixed with synthetic data.
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "biasbios_train.csv"
    val_path = processed_dir / "biasbios_val.csv"

    # Save only the canonical columns used downstream to ensure no nulls
    # appear due to source-specific metadata.
    train_save = train_df[["text", "label"]].copy()
    val_save = val_df[["text", "label"]].copy()

    # Ensure dtypes are consistent
    train_save["text"] = train_save["text"].astype(str)
    train_save["label"] = pd.to_numeric(train_save["label"], errors="coerce").astype(
        int
    )
    val_save["text"] = val_save["text"].astype(str)
    val_save["label"] = pd.to_numeric(val_save["label"], errors="coerce").astype(int)

    train_save.to_csv(train_path, index=False)
    val_save.to_csv(val_path, index=False)

    logger.info("Saved preprocessed training set to: %s", train_path)
    logger.info("Saved preprocessed validation set to: %s", val_path)


if __name__ == "__main__":
    main()

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import t, ttest_rel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# ================================================================
# USER CONFIG (EDIT THESE or pass via CLI)
# ================================================================
# By default this script uses the repository-local paths. You can
# override any of these via the command-line args (see --help).
TEST_CSV = "data/processed/biasbios_test.csv"  # default test dataset in repo
TEXT_COL = "text"
LABEL_COL = "label_binary"

# Sensitive attribute columns (e.g., ["gender", "race"]). Keep empty to skip fairness.
SENSITIVE_COLS: List[str] = []

# Model directories / files (repo-local defaults)
BIASGUARD_DIR = "models"  # BiasGuard Pro model + tokenizer files (models/)
ROBERTA_DIR = "models/baselines/roberta_baseline"
CDA_DIR = "models/baselines/cda"

# TF-IDF Logistic artifacts
TFIDF_MODEL = "models/baselines/tfidf_logreg_model.joblib"
TFIDF_VECTORIZER = "models/baselines/tfidf_logreg_vectorizer.joblib"
TFIDF_PREDICTIONS_CSV = "results/tfidf_logreg_predictions.csv"

# GloVe SVM artifacts / CSV
GLOVE_SVM_MODEL = "models/baselines/glove_svm_model.joblib"
GLOVE_SVM_PREDICTIONS_CSV = "results/glove_svm_predictions.csv"

# Output directory (place results in the repo results/ folder)
OUT_DIR = "results"

# Inference settings (can be tuned)
BATCH_SIZE = 32
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


# ================================================================
# Utility Functions
# ================================================================
def load_test_data(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    """Load test CSV and validate required columns.

    Args:
        path: Path to CSV file.
        text_col: Expected text column name.
        label_col: Expected label column name.

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """

    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Missing required columns '{text_col}' or '{label_col}' in {path}"
        )
    return df


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Compute row-wise softmax for a 2D NumPy array of logits.

    Args:
        logits: Array of shape (n_samples, n_classes).

    Returns:
        Array of same shape with probabilities summing to 1 per row.
    """

    e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


class TextDataset(Dataset):
    """Lightweight dataset wrapper used for HF model inference.

    This dataset yields raw text strings and exposes a `collate_fn` that
    performs tokenization into tensors when passed to a DataLoader.
    """

    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

    def collate_fn(self, batch):
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


def try_load_hf_model(model_dir: str) -> Optional[Tuple[Any, Any]]:
    """Attempt to load a HuggingFace tokenizer and model from `model_dir`.

    Returns (tokenizer, model) on success or None on failure. Errors are
    surfaced via a printed warning to keep behavior lightweight in CLI runs.
    """

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(DEVICE).eval()
        return tokenizer, model
    except Exception as e:
        print(f"[WARN] Could not load HF model from {model_dir}: {e}")
        return None


def robust_load_roberta(model_dir: str) -> Optional[Tuple[Any, Any]]:
    """
    Load a local RoBERTa checkpoint even if config.json is missing or
    positional embedding size mismatch occurs (514 vs 512).
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        print(f"[WARN] RoBERTa tokenizer load failed: {e}")
        return None

    # Try config from directory; fallback to roberta-base
    try:
        config = AutoConfig.from_pretrained(model_dir)
    except Exception:
        config = AutoConfig.from_pretrained("roberta-base")

    # Force binary classification unless specified
    if getattr(config, "num_labels", None) != 2:
        config.num_labels = 2

    # Standard RoBERTa uses 514 position embeddings
    if getattr(config, "max_position_embeddings", 0) < 514:
        config.max_position_embeddings = 514

    # Align vocab size with tokenizer
    vocab_size = (
        len(tokenizer)
        if hasattr(tokenizer, "__len__")
        else getattr(config, "vocab_size", None)
    )
    if vocab_size and getattr(config, "vocab_size", None) != vocab_size:
        config.vocab_size = vocab_size

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, config=config, ignore_mismatched_sizes=True
        )
        model.to(DEVICE).eval()
        return tokenizer, model
    except Exception as e:
        print(f"[WARN] RoBERTa model load failed: {e}")
        return None


@torch.no_grad()
def predict_hf(model, tokenizer, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Run batched inference for a HuggingFace sequence classification model.

    Args:
        model: HF model instance.
        tokenizer: Corresponding tokenizer.
        texts: List of input strings.

    Returns:
        Tuple of (preds: ndarray[int], proba_pos: ndarray[float]) where preds is
        the binary predicted labels and proba_pos are the positive-class probs.
    """

    ds = TextDataset(texts, tokenizer, max_length=MAX_LENGTH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ds.collate_fn)
    all_logits = []
    for batch in tqdm(dl, desc="HF Inference", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        logits = out.logits.detach().cpu().numpy()
        all_logits.append(logits)
    logits = np.vstack(all_logits)
    if logits.shape[1] == 1:
        probs_pos = 1 / (1 + np.exp(-logits[:, 0]))
        probs = np.vstack([1 - probs_pos, probs_pos]).T
    else:
        probs = softmax_np(logits)
    proba_pos = probs[:, 1]
    preds = (proba_pos >= 0.5).astype(int)
    return preds, proba_pos


def try_load_sklearn_model(model_path: str, vectorizer_path: Optional[str] = None):
    """Load a scikit-learn model and optional vectorizer from disk.

    Returns (model, vectorizer) or (None, None) on failure.
    """

    try:
        model = joblib.load(model_path)
        vect = (
            joblib.load(vectorizer_path)
            if vectorizer_path and Path(vectorizer_path).exists()
            else None
        )
        return model, vect
    except Exception as e:
        print(f"[WARN] Sklearn model load failed ({model_path}): {e}")
        return None, None


def predict_sklearn(
    model, vectorizer, texts: List[str]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Predict using a scikit-learn model, optionally using a vectorizer.

    Returns a tuple (y_pred, proba) where proba may be None if unavailable.
    """

    if vectorizer is not None:
        X = vectorizer.transform(texts)
    else:
        # If model is a pipeline it can handle raw texts; else error
        try:
            X = texts
        except Exception:
            raise ValueError("No vectorizer and model not a pipeline.")
    y_pred = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] == 2:
            proba = p[:, 1]
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X)
        if df.ndim == 1:
            proba = 1 / (1 + np.exp(-df))
    return np.array(y_pred).astype(int), proba


def read_predictions_csv(pred_csv: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read a predictions CSV and return prediction and optional proba arrays.

    The function attempts a set of common column names for predictions and
    probabilities and falls back to detecting a single binary column when
    needed.
    """

    df = pd.read_csv(pred_csv)
    pred_col = next(
        (
            c
            for c in ["pred", "prediction", "y_pred", "label_pred", "predicted_label"]
            if c in df.columns
        ),
        None,
    )
    if pred_col is None:
        # fallback: single binary column?
        bin_cols = [c for c in df.columns if df[c].dropna().isin([0, 1]).all()]
        if len(bin_cols) == 1:
            pred_col = bin_cols[0]
        else:
            raise ValueError(f"No prediction column found in {pred_csv}")
    proba_col = next(
        (
            c
            for c in ["proba", "prob", "prob_pos", "probability", "score"]
            if c in df.columns
        ),
        None,
    )
    y_pred = df[pred_col].astype(int).to_numpy()
    y_proba = df[proba_col].to_numpy() if proba_col else None
    return y_pred, y_proba


def ensure_len_match(y_true: np.ndarray, arr: np.ndarray, name: str):
    """Ensure two arrays have matching lengths; raise ValueError otherwise.

    Args:
        y_true: Reference array.
        arr: Array to compare.
        name: Human-readable name used in the error message.
    """

    if len(y_true) != len(arr):
        raise ValueError(f"Length mismatch for {name}: {len(arr)} vs {len(y_true)}")


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]
) -> Dict[str, Any]:
    """Compute core classification metrics (accuracy, precision, recall, f1)

    If probability estimates are available, also compute ROC AUC when
    possible.
    """

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out = dict(accuracy=acc, precision=prec, recall=rec, f1=f1)
    if y_proba is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            out["roc_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan
    return out


def mcnemar_test(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
) -> Dict[str, Any]:
    """Run McNemar's test comparing two sets of predictions A and B.

    Returns a dict with contingency counts, test statistic, p-value and
    effect-size summaries (Cohen's g and odds ratio when defined).
    """

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    b = int(((correct_a == True) & (correct_b == False)).sum())  # A correct, B wrong
    c = int(((correct_a == False) & (correct_b == True)).sum())  # A wrong, B correct
    table = [[0, b], [c, 0]]
    exact = (b + c) < 25
    result = mcnemar(table, exact=exact, correction=not exact)
    n = len(y_true)
    cohens_g = abs(b - c) / n if n else np.nan
    odds_ratio = (b / c) if c != 0 else (math.inf if b > 0 else np.nan)
    return {
        "b_A_correct_B_wrong": b,
        "c_A_wrong_B_correct": c,
        "exact": exact,
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "cohens_g": cohens_g,
        "odds_ratio": odds_ratio,
    }


def paired_t_test_trueclass_proba(
    y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray
) -> Dict[str, Any]:
    """Paired t-test on per-example 'true-class' probabilities.

    Converts probability estimates to the probability assigned to the true
    class for each example, then performs a paired t-test between two
    systems' scores. Returns t-statistic, p-value, Cohen's dz and a 95% CI.
    """

    score_a = np.where(y_true == 1, proba_a, 1 - proba_a)
    score_b = np.where(y_true == 1, proba_b, 1 - proba_b)
    diffs = score_b - score_a
    t_stat, p_val = ttest_rel(score_b, score_a, nan_policy="omit")
    sd = np.nanstd(diffs, ddof=1)
    mean_diff = np.nanmean(diffs)
    n = np.sum(~np.isnan(diffs))
    cohens_dz = mean_diff / sd if sd and sd > 0 else np.nan
    if n > 1 and sd and sd > 0:
        se = sd / np.sqrt(n)
        t_crit = t.ppf(0.975, df=n - 1)
        ci_low = mean_diff - t_crit * se
        ci_high = mean_diff + t_crit * se
    else:
        ci_low = np.nan
        ci_high = np.nan
    return {
        "t_statistic": t_stat,
        "p_value": p_val,
        "cohens_dz": cohens_dz,
        "mean_diff_trueclass_proba": mean_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "n": int(n),
    }


def compute_fairness(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive: pd.Series
) -> Dict[str, Any]:
    """Compute simple group fairness metrics across sensitive groups.

    Returns per-group counts and positive-rate / TPR / FPR / precision
    summaries plus overall gap and ratio statistics for common fairness axes.
    """

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "s": sensitive})
    groups = df["s"].dropna().unique()
    group_metrics = {}
    pos_rates, tprs, fprs, precs = [], [], [], []

    for g in groups:
        sub = df[df["s"] == g]
        yt = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()
        pr = (yp == 1).mean() if len(yp) else np.nan
        tpr = recall_score(yt, yp, pos_label=1, zero_division=0) if len(yp) else np.nan
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            fpr = 0.0
        prec = (
            precision_score(yt, yp, pos_label=1, zero_division=0) if len(yp) else np.nan
        )

        pos_rates.append(pr)
        tprs.append(tpr)
        fprs.append(fpr)
        precs.append(prec)
        group_metrics[g] = {
            "count": int(len(sub)),
            "positive_rate": pr,
            "tpr": tpr,
            "fpr": fpr,
            "precision": prec,
        }

    def gap_ratio(values):
        arr = np.array(values, dtype=float)
        if len(arr) == 0 or np.all(np.isnan(arr)):
            return np.nan, np.nan
        vmax = np.nanmax(arr)
        vmin = np.nanmin(arr)
        ratio = (vmin / vmax) if vmax > 0 else np.nan
        return float(vmax - vmin), float(ratio)

    pr_gap, pr_ratio = gap_ratio(pos_rates)
    tpr_gap, tpr_ratio = gap_ratio(tprs)
    fpr_gap, fpr_ratio = gap_ratio(fprs)
    prec_gap, prec_ratio = gap_ratio(precs)

    return {
        "groups": group_metrics,
        "demographic_parity": {"gap": pr_gap, "ratio": pr_ratio},
        "equal_opportunity_tpr": {"gap": tpr_gap, "ratio": tpr_ratio},
        "fpr_parity": {"gap": fpr_gap, "ratio": fpr_ratio},
        "predictive_parity_precision": {"gap": prec_gap, "ratio": prec_ratio},
    }


def build_baselines(
    tfidf_model: str,
    tfidf_vect: str,
    tfidf_csv: str,
    glove_model: str,
    glove_csv: str,
    roberta_dir: str,
    cda_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Construct the baseline registry from available artifacts.

    Returns a dict where each key is a baseline name and value is a config dict
    with keys: type (hf|sklearn|csv), and the associated artifacts.
    """
    baselines: Dict[str, Dict[str, Any]] = {}

    # Robust RoBERTa
    if Path(roberta_dir).exists():
        roberta_loaded = robust_load_roberta(roberta_dir)
        if roberta_loaded:
            baselines["roberta"] = {
                "type": "hf",
                "tokenizer": roberta_loaded[0],
                "model": roberta_loaded[1],
            }
        else:
            print("[INFO] RoBERTa skipped (robust load failed).")
    else:
        print("[INFO] RoBERTa directory missing; skipped.")

    # CDA HF
    if Path(cda_dir).exists():
        cda_loaded = try_load_hf_model(cda_dir)
        if cda_loaded:
            baselines["cda"] = {
                "type": "hf",
                "tokenizer": cda_loaded[0],
                "model": cda_loaded[1],
            }
        else:
            print("[INFO] CDA skipped (load failed).")
    else:
        print("[INFO] CDA directory missing; skipped.")

    # TF-IDF Logistic (prefer model; fallback CSV)
    if Path(tfidf_model).exists() and Path(tfidf_vect).exists():
        tfidf_m, tfidf_v = try_load_sklearn_model(tfidf_model, tfidf_vect)
        if tfidf_m is not None:
            baselines["tfidf_logreg"] = {
                "type": "sklearn",
                "model": tfidf_m,
                "vectorizer": tfidf_v,
            }
    elif Path(tfidf_csv).exists():
        baselines["tfidf_logreg"] = {"type": "csv", "pred_csv": tfidf_csv}
    else:
        print("[INFO] TF-IDF Logistic unavailable; skipped.")

    # GloVe SVM (prefer CSV)
    if Path(glove_csv).exists():
        baselines["glove_svm"] = {"type": "csv", "pred_csv": glove_csv}
    elif Path(glove_model).exists():
        glove_m, glove_v = try_load_sklearn_model(glove_model, None)
        if glove_m is not None:
            baselines["glove_svm"] = {
                "type": "sklearn",
                "model": glove_m,
                "vectorizer": getattr(glove_m, "vectorizer_", glove_v),
            }
        else:
            print("[INFO] GloVe SVM joblib failed; skipped.")
    else:
        print("[INFO] GloVe SVM artifacts missing; skipped.")

    return baselines


def run_evaluation(
    test_csv: str,
    text_col: str,
    label_col: str,
    biasguard_dir: str,
    baselines: Dict[str, Dict[str, Any]],
    sensitive_cols: List[str],
    out_dir: str,
) -> None:
    """Main evaluation: loads test data, runs BiasGuard and baselines,
    computes paired tests and fairness metrics, then writes results to out_dir.
    """
    df_test = load_test_data(test_csv, text_col, label_col)
    texts = df_test[text_col].astype(str).tolist()
    y_true = df_test[label_col].astype(int).to_numpy()

    # Load BiasGuard Pro (reference model)
    biasguard_loaded = try_load_hf_model(biasguard_dir)
    if biasguard_loaded is None:
        raise RuntimeError(
            f"BiasGuard Pro failed to load from {biasguard_dir}. Check directory and files."
        )
    bg_tokenizer, bg_model = biasguard_loaded
    print("Running inference for BiasGuard Pro...")
    bg_pred, bg_proba = predict_hf(bg_model, bg_tokenizer, texts)
    ensure_len_match(y_true, bg_pred, "BiasGuard preds")
    ensure_len_match(y_true, bg_proba, "BiasGuard proba")
    bg_metrics = classification_metrics(y_true, bg_pred, bg_proba)
    print("BiasGuard metrics:", bg_metrics)

    # Prepare evaluation containers
    rows_summary: List[Dict[str, Any]] = []
    per_model_metrics: Dict[str, Dict[str, Any]] = {}

    def evaluate_baseline_local(
        name: str, cfg: Dict[str, Any]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if cfg["type"] == "hf":
            print(f"Running inference for {name} (HF)...")
            return predict_hf(cfg["model"], cfg["tokenizer"], texts)
        elif cfg["type"] == "sklearn":
            print(f"Running inference for {name} (sklearn)...")
            return predict_sklearn(cfg["model"], cfg.get("vectorizer"), texts)
        elif cfg["type"] == "csv":
            print(f"Loading predictions for {name} (CSV)...")
            y_pred_b, y_proba_b = read_predictions_csv(cfg["pred_csv"])
            ensure_len_match(y_true, y_pred_b, f"{name} preds CSV")
            if y_proba_b is not None:
                ensure_len_match(y_true, y_proba_b, f"{name} proba CSV")
            return y_pred_b, y_proba_b
        else:
            raise ValueError(f"Unknown baseline type: {cfg['type']}")

    for base_name, base_cfg in baselines.items():
        try:
            y_pred_b, y_proba_b = evaluate_baseline_local(base_name, base_cfg)
        except Exception as e:
            print(f"[WARN] Baseline {base_name} evaluation failed: {e}")
            continue

        metrics_b = classification_metrics(y_true, y_pred_b, y_proba_b)
        per_model_metrics[base_name] = metrics_b

        # McNemar (baseline vs BiasGuard)
        mcn = mcnemar_test(y_true, y_pred_b, bg_pred)

        # Paired t-test only if both have probabilities
        if y_proba_b is not None and bg_proba is not None:
            try:
                ttest_res = paired_t_test_trueclass_proba(y_true, y_proba_b, bg_proba)
            except Exception as e:
                print(f"[INFO] Paired t-test failed for {base_name}: {e}")
                ttest_res = None
        else:
            ttest_res = None

        # Note: fairness metrics generation has been disabled per configuration.

        row = {
            "baseline": base_name,
            "baseline_accuracy": metrics_b["accuracy"],
            "baseline_precision": metrics_b["precision"],
            "baseline_recall": metrics_b["recall"],
            "baseline_f1": metrics_b["f1"],
            "baseline_roc_auc": metrics_b.get("roc_auc", np.nan),
            "biasguard_accuracy": bg_metrics["accuracy"],
            "biasguard_precision": bg_metrics["precision"],
            "biasguard_recall": bg_metrics["recall"],
            "biasguard_f1": bg_metrics["f1"],
            "biasguard_roc_auc": bg_metrics.get("roc_auc", np.nan),
            "mcnemar_b_Acorrect_Bwrong": mcn["b_A_correct_B_wrong"],
            "mcnemar_c_Awrong_Bcorrect": mcn["c_A_wrong_B_correct"],
            "mcnemar_exact": mcn["exact"],
            "mcnemar_statistic": mcn["statistic"],
            "mcnemar_p_value": mcn["p_value"],
            "mcnemar_cohens_g": mcn["cohens_g"],
            "mcnemar_odds_ratio": mcn["odds_ratio"],
        }
        if ttest_res:
            row.update(
                {
                    "paired_t_stat": ttest_res["t_statistic"],
                    "paired_t_p_value": ttest_res["p_value"],
                    "paired_t_cohens_dz": ttest_res["cohens_dz"],
                    "paired_t_mean_diff_trueclass_proba": ttest_res[
                        "mean_diff_trueclass_proba"
                    ],
                    "paired_t_ci95_low": ttest_res["ci95_low"],
                    "paired_t_ci95_high": ttest_res["ci95_high"],
                    "paired_t_n": ttest_res["n"],
                }
            )
        else:
            row.update(
                {
                    "paired_t_stat": np.nan,
                    "paired_t_p_value": np.nan,
                    "paired_t_cohens_dz": np.nan,
                    "paired_t_mean_diff_trueclass_proba": np.nan,
                    "paired_t_ci95_low": np.nan,
                    "paired_t_ci95_high": np.nan,
                    "paired_t_n": np.nan,
                }
            )
        rows_summary.append(row)

    # Save outputs to out_dir
    os.makedirs(out_dir, exist_ok=True)
    summary_df = pd.DataFrame(rows_summary)
    summary_path = os.path.join(out_dir, "paired_tests_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved paired tests summary -> {summary_path}")

    metrics_rows = [{"model": "biasguard_pro", **bg_metrics}]
    for model_name, met in per_model_metrics.items():
        metrics_rows.append({"model": model_name, **met})
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(out_dir, "classification_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved classification metrics -> {metrics_path}")

    # Fairness metrics output intentionally omitted.

    print("\n=== Paired Tests Summary Preview ===")
    print(summary_df.head())
    print("\n=== Classification Metrics Preview ===")
    print(metrics_df.head())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run paired statistical tests between BiasGuard and baselines."
    )
    p.add_argument("--test-csv", default=TEST_CSV, help="Path to test CSV")
    p.add_argument("--text-col", default=TEXT_COL, help="Text column name in test CSV")
    p.add_argument(
        "--label-col", default=LABEL_COL, help="Label column name in test CSV"
    )
    p.add_argument(
        "--biasguard-dir", default=BIASGUARD_DIR, help="BiasGuard model directory"
    )
    p.add_argument("--roberta-dir", default=ROBERTA_DIR, help="RoBERTa baseline dir")
    p.add_argument("--cda-dir", default=CDA_DIR, help="CDA baseline dir")
    p.add_argument(
        "--tfidf-model", default=TFIDF_MODEL, help="TF-IDF model joblib path"
    )
    p.add_argument(
        "--tfidf-vect", default=TFIDF_VECTORIZER, help="TF-IDF vectorizer joblib path"
    )
    p.add_argument(
        "--tfidf-csv",
        default=TFIDF_PREDICTIONS_CSV,
        help="TF-IDF predictions CSV (fallback)",
    )
    p.add_argument(
        "--glove-model", default=GLOVE_SVM_MODEL, help="GloVe SVM model joblib path"
    )
    p.add_argument(
        "--glove-csv",
        default=GLOVE_SVM_PREDICTIONS_CSV,
        help="GloVe predictions CSV (preferred)",
    )
    p.add_argument("--out-dir", default=OUT_DIR, help="Output directory for results")
    p.add_argument(
        "--sensitive-cols",
        nargs="*",
        default=SENSITIVE_COLS,
        help="Sensitive columns to compute fairness on",
    )
    return p.parse_args()


def main():
    args = parse_args()

    baselines = build_baselines(
        tfidf_model=args.tfidf_model,
        tfidf_vect=args.tfidf_vect,
        tfidf_csv=args.tfidf_csv,
        glove_model=args.glove_model,
        glove_csv=args.glove_csv,
        roberta_dir=args.roberta_dir,
        cda_dir=args.cda_dir,
    )

    run_evaluation(
        test_csv=args.test_csv,
        text_col=args.text_col,
        label_col=args.label_col,
        biasguard_dir=args.biasguard_dir,
        baselines=baselines,
        sensitive_cols=args.sensitive_cols,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

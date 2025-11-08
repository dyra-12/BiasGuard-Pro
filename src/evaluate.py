"""Unified evaluation entrypoint.

This script evaluates all available trained models and saves ONLY a
classification report CSV per model under `results/`.

Supported models and expected artifacts:
 - HF (transformers) directories:
     * models/ (top-level trained model)
     * models/baselines/roberta_baseline/
     * models/baselines/cda/
 - Classical baselines:
     * models/baselines/tfidf_logreg_vectorizer.joblib + tfidf_logreg_model.joblib
     * models/baselines/glove_svm_model.joblib (+ optional embeddings .kv)
"""

from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # will gate classical baselines if missing

try:
    from gensim.models import KeyedVectors  # type: ignore
except Exception:  # pragma: no cover
    KeyedVectors = None

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ------------------------- data helpers -------------------------
def _detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)


def load_biasbios_test(processed_dir: Path = Path("data/processed")) -> Tuple[List[str], List[int]]:
    test_path = processed_dir / "biasbios_test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")
    df = pd.read_csv(test_path)
    text_col = _detect_column(df, ["text", "cleaned_text", "hard_text", "context", "sentence"]) or "text"
    label_col = _detect_column(df, ["label", "label_binary", "bias_label", "binary_label"]) or "label"
    texts = df[text_col].astype(str).fillna("").tolist()
    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).tolist()
    return texts, labels


def save_classification_report_csv(report_dict: Dict, out_path: Path) -> None:
    rows = []
    for key, vals in report_dict.items():
        if isinstance(vals, dict):
            row = {"label": key}
            row.update(vals)
            rows.append(row)
        else:
            rows.append({"label": key, "value": vals})
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ------------------------- model evaluators -------------------------
def eval_transformer_model(model_dir: Path, texts: List[str], labels: List[int]) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    all_preds: List[int] = []
    with torch.no_grad():
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            all_preds.extend(preds)

    report = classification_report(labels, all_preds, output_dict=True, zero_division=0)
    return report


def eval_tfidf_logreg(models_dir: Path, texts: List[str], labels: List[int]) -> Optional[Dict]:
    if joblib is None:
        logger.warning("joblib not available; skipping tfidf_logreg evaluation")
        return None
    vec_path = models_dir / "tfidf_logreg_vectorizer.joblib"
    mdl_path = models_dir / "tfidf_logreg_model.joblib"
    if not vec_path.exists() or not mdl_path.exists():
        return None
    tfidf = joblib.load(vec_path)
    clf = joblib.load(mdl_path)
    X = tfidf.transform(texts)
    preds = clf.predict(X)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return report


def eval_glove_svm(models_dir: Path, texts: List[str], labels: List[int]) -> Optional[Dict]:
    if joblib is None:
        logger.warning("joblib not available; skipping glove_svm evaluation")
        return None
    mdl_path = models_dir / "glove_svm_model.joblib"
    if not mdl_path.exists():
        return None
    clf = joblib.load(mdl_path)

    # Attempt to load embeddings (if gensim and kv file available)
    kv_path = models_dir / "glove-wiki-gigaword-300.kv"
    if KeyedVectors is None or not kv_path.exists():
        logger.warning("Embeddings not available for glove_svm; skipping evaluation")
        return None
    kv = KeyedVectors.load(str(kv_path))
    emb_dim = int(getattr(kv, "vector_size", 300))

    def doc_to_vec(text: str) -> np.ndarray:
        words = str(text).split()
        vecs = [kv[w] for w in words if w in kv.key_to_index]
        if vecs:
            return np.mean(vecs, axis=0)
        return np.zeros((emb_dim,), dtype=np.float32)

    X = np.vstack([doc_to_vec(t) for t in texts])
    preds = clf.predict(X)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return report


# ------------------------- discovery & driver -------------------------
def discover_models() -> Dict[str, Path]:
    """Return mapping of model_name -> path/artifacts root to evaluate."""
    registry: Dict[str, Path] = {}

    # Transformers dirs
    hf_main = Path("models")
    if (hf_main / "config.json").exists():
        registry["transformer_main"] = hf_main

    rb = Path("models/baselines/roberta_baseline")
    if (rb / "config.json").exists():
        registry["roberta_baseline"] = rb

    cda = Path("models/baselines/cda")
    if (cda / "config.json").exists():
        registry["cda"] = cda

    # Classical baselines
    baselines_dir = Path("models/baselines")
    if (baselines_dir / "tfidf_logreg_model.joblib").exists():
        registry["tfidf_logreg"] = baselines_dir
    if (baselines_dir / "glove_svm_model.joblib").exists():
        registry["glove_svm"] = baselines_dir

    return registry


def main():
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    texts, labels = load_biasbios_test()
    registry = discover_models()
    if not registry:
        logger.warning("No models discovered to evaluate.")
        return

    logger.info("Discovered models: %s", ", ".join(registry.keys()))

    for name, path in registry.items():
        try:
            logger.info("Evaluating %s ...", name)
            if name in ("roberta_baseline", "cda", "transformer_main"):
                report = eval_transformer_model(path, texts, labels)
            elif name == "tfidf_logreg":
                report = eval_tfidf_logreg(path, texts, labels)
            elif name == "glove_svm":
                report = eval_glove_svm(path, texts, labels)
            else:
                logger.warning("Unknown model type for %s; skipping", name)
                report = None

            if report is None:
                logger.warning("Skipping %s due to missing artifacts or dependencies", name)
                continue

            out_csv = results_dir / f"classification_report_{name}.csv"
            save_classification_report_csv(report, out_csv)
            logger.info("Saved classification report -> %s", out_csv)
        except Exception as e:
            logger.exception("Failed to evaluate %s: %s", name, e)


if __name__ == "__main__":
    main()


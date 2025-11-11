import sys
from pathlib import Path
import pandas as pd
import numpy as np

from importlib.util import spec_from_file_location, module_from_spec


def load_module(name: str, rel_path: str):
    """Load a module from the repo's src/ directory by relative path.

    This avoids package import issues while running tests from the repo root.
    """
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "src" / rel_path
    spec = spec_from_file_location(name, str(mod_path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_biasbios_tmp_csv(tmp_path):
    evaluate = load_module("evaluate", "evaluate.py")
    # create a tiny test csv
    df = pd.DataFrame({"text": ["alice is an engineer", "bob is a nurse"], "label": [0, 1]})
    p = tmp_path / "biasbios_test.csv"
    df.to_csv(p, index=False)

    texts, labels = evaluate.load_biasbios_test(processed_dir=tmp_path)
    assert isinstance(texts, list) and isinstance(labels, list)
    assert texts == ["alice is an engineer", "bob is a nurse"]
    assert labels == [0, 1]


def test_classification_and_stat_tests():
    # Load stats_tests module directly
    stats = load_module("stats_tests", "eval/stats_tests.py")

    # classification_metrics basic sanity
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2])
    out = stats.classification_metrics(y_true, y_pred, y_proba)
    assert "accuracy" in out and "f1" in out
    assert abs(out["accuracy"] - 0.75) < 1e-6

    # McNemar test simple case
    y_true2 = np.array([0, 1, 0, 1])
    y_pred_a = np.array([0, 1, 1, 1])
    y_pred_b = np.array([1, 1, 0, 0])
    m = stats.mcnemar_test(y_true2, y_pred_a, y_pred_b)
    # Expect b = 2 (A correct, B wrong at idx 0 and 3), c = 1 (A wrong, B correct at idx 2)
    assert m["b_A_correct_B_wrong"] == 2
    assert m["c_A_wrong_B_correct"] == 1

    # Paired t-test on true-class probabilities
    y_true3 = np.array([1, 0, 1, 0])
    proba_a = np.array([0.6, 0.2, 0.7, 0.1])
    proba_b = np.array([0.7, 0.3, 0.8, 0.2])
    tres = stats.paired_t_test_trueclass_proba(y_true3, proba_a, proba_b)
    assert "p_value" in tres and "t_statistic" in tres
    assert tres["n"] == 4

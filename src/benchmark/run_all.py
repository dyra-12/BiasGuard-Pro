"""Unified experiment runner for BiasGuard Pro.

This script provides a single entrypoint to reproduce the core
experiments and tables reported in the paper/README:

Stages:
  1. model_reports   -> per-model classification reports (evaluate.py)
  2. cross_dataset   -> BiasBios vs StereoSet evaluation (cross_dataset_evaluation.py)
  3. stats_tests     -> paired statistical tests & metrics (eval/stats_tests.py)

By default it runs all three stages ("--stage all"). It also accepts
legacy-style arguments requested in the README template (e.g.
"--models all --dataset biasbios_test --output results/") for
compatibility; these are mapped internally to the new flags.

Example:
  python src/benchmark/run_all.py --stage all
  python src/benchmark/run_all.py --stage stats_tests

The outputs are written under `results/`:
  classification_report_*.csv
  crossdataset_results.csv
  paired_tests_summary.csv
  classification_metrics.csv

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import importlib

RESULTS_DIR = Path("results")


def run_model_reports():
	"""Invoke evaluate.py to generate per-model classification reports.

	evaluate.py already discovers models and writes CSVs to results/.
	"""
	print("[run_all] Stage: model_reports")
	mod = importlib.import_module("src.evaluate".replace("/", ".").replace(".py", "")) if False else None
	# Direct import using package-style relative path fallback
	try:
		import evaluate as eval_mod  # type: ignore
	except ImportError as e:
		print(f"[WARN] Could not import evaluate.py directly: {e}")
		raise
	eval_mod.main()


def run_cross_dataset(model_path: str):
	"""Invoke cross_dataset_evaluation.py for BiasBios + StereoSet."""
	print("[run_all] Stage: cross_dataset")
	try:
		import cross_dataset_evaluation as cde  # type: ignore
	except ImportError:
		from src import cross_dataset_evaluation as cde  # type: ignore

	# Use defaults inside module via argument emulation
	sys.argv = [
		"cross_dataset_evaluation.py",
		"--model_path", model_path,
	]  # rely on internal defaults for dataset paths
	cde.main()


def run_stats_tests():
	"""Invoke stats_tests.py for paired tests and aggregate metrics."""
	print("[run_all] Stage: stats_tests")
	try:
		from src.eval import stats_tests as st  # type: ignore
	except ImportError:
		import stats_tests as st  # type: ignore
	st.main()


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Unified experiment runner for BiasGuard Pro")
	p.add_argument("--stage", choices=["all", "model_reports", "cross_dataset", "stats_tests"], default="all",
				   help="Which stage to run (default: all)")
	# Legacy / compatibility flags (ignored but accepted)
	p.add_argument("--models", default="all", help="(compat) Models spec; ignored (auto-discovery)")
	p.add_argument("--dataset", default="biasbios_test", help="(compat) Dataset name; internal scripts use fixed paths")
	p.add_argument("--output", default=str(RESULTS_DIR), help="(compat) Output directory; scripts write to results/")
	p.add_argument("--model-path", default="models", help="Path to main BiasGuard model directory")
	return p.parse_args()


def main():
	args = parse_args()
	RESULTS_DIR.mkdir(exist_ok=True, parents=True)

	if args.stage in ("all", "model_reports"):
		run_model_reports()
	if args.stage in ("all", "cross_dataset"):
		run_cross_dataset(args.model_path)
	if args.stage in ("all", "stats_tests"):
		run_stats_tests()

	print("[run_all] Done.")


if __name__ == "__main__":
	main()


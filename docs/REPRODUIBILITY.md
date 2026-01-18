# Reproducibility Guide (BiasGuard Pro)

This document describes how to reproduce the main experimental artifacts in this repository, at three levels of rigor:

1. **Artifact-based reproduction (fast, recommended):** re-run evaluation/analysis from the shipped checkpoints + processed data.
2. **Data regeneration:** re-download raw datasets and regenerate the processed splits used by the code.
3. **Model re-training (advanced):** re-train baselines (and optionally re-train the BiasGuard model) and then re-run the full evaluation suite.

The repository is designed so that Level 1 is feasible on CPU-only machines, while Level 3 may require a GPU and significant runtime.

---

## 0. Reproducibility Metadata

Record the exact code + environment used for your run.

- **Repository:** BiasGuard-Pro
- **Commit:** `515749cef1d5882a9cdbe0afa40709927cc9a53a`
- **Working tree:** may be dirty (run `git status --porcelain` to confirm)
- **Primary model artifact:** `models/model.safetensors`
- **Model version tag:** `models/model_version.txt` (currently: `placeholder_v0.1.0`)

Recommended to capture:

```bash
python --version
pip --version
pip freeze > reproducibility_pip_freeze.txt
uname -a  # Linux
git rev-parse HEAD
git status --porcelain
```

---

## 1. Environment Setup

### 1.1 Python

- **Python:** 3.10+

Create an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- Dependencies are pinned in `requirements.txt` for reproducibility.
- The shipped BiasGuard checkpoint records a training-time `transformers_version` in `models/config.json`. If you need strict environment matching, align your `transformers` version to that value; otherwise, inference typically remains compatible across nearby versions.
- GPU acceleration is optional. If you use CUDA, ensure your local PyTorch build matches your CUDA runtime.

### 1.2 Determinism / Seeding

This codebase uses fixed seeds in some scripts (e.g., `src/eval/stats_tests.py` uses `SEED = 42`). However, exact bitwise reproducibility is not guaranteed on GPU due to nondeterministic kernels.

For best-effort determinism:

```bash
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
```

If you control your own training runs, consider enabling deterministic algorithms in PyTorch and disabling cuDNN benchmarking (may reduce speed):

```python
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

---

## 2. What Is Already Included in the Repository

### 2.1 Data

This repository ships with:

- **Processed splits:** `data/processed/biasbios_train.csv`, `data/processed/biasbios_val.csv`, `data/processed/biasbios_test.csv`
- **Raw StereoSet gender subset:** `data/raw/stereoset_gender.csv`
- **Raw BiasBios master CSV:** `data/raw/master_dataset.csv`
- **Synthetic paired examples:** `data/raw/synthetic_career_data.csv`

Level 1 reproduction does **not** require re-downloading datasets.

### 2.2 Models

This repository ships with:

- BiasGuard Pro (DistilBERT) checkpoint under `models/`
- Baselines under `models/baselines/` (TF-IDF Logistic Regression, GloVe+SVM, RoBERTa baseline, CDA baseline)

### 2.3 Results

The `results/` directory includes precomputed artifacts used in the README and plots, including:

- `results/crossdataset_results.csv`
- `results/paired_tests_summary.csv`
- `results/*_predictions.csv`
- `results/metrics_per_model.csv` (used for plotting figures)

---

## 3. Level 1 — Re-run Main Experiments (Recommended)

This level assumes you will use the shipped datasets and checkpoints.

### 3.1 One-command reproduction

Run the unified experiment harness:

```bash
python src/benchmark/run_all.py --stage all
```

Expected outputs (written under `results/`):

- Classification reports from `src/evaluate.py` (one CSV per model)
- Cross-dataset evaluation from `src/cross_dataset_evaluation.py`:
	- `biasbios_test_predictions.csv`, `stereoset_gender_predictions.csv`
	- `crossdataset_results.csv`
- Paired statistical tests from `src/eval/stats_tests.py`:
	- `paired_tests_summary.csv`
	- `classification_metrics.csv`

### 3.2 Run stages individually

```bash
# Per-model classification reports
python src/evaluate.py

# Cross-dataset evaluation (BiasBios test vs StereoSet gender)
python src/cross_dataset_evaluation.py \
	--model_path models \
	--biasbios_path data/processed/biasbios_test.csv \
	--stereoset_path data/raw/stereoset_gender.csv

# Statistical tests and metrics aggregation
python src/eval/stats_tests.py \
	--test-csv data/processed/biasbios_test.csv \
	--text-col text \
	--label-col label_binary
```

Tip: if your processed test split uses `label` rather than `label_binary`, pass `--label-col label`.

---

## 4. Level 2 — Regenerate Data Locally

This level re-downloads raw data and re-creates `data/processed/`.

### 4.1 Re-download raw datasets (BiasBios + StereoSet)

The automated downloader is:

- `data/download_datasets.py`

Run:

```bash
python data/download_datasets.py
```

Outputs:

- `data/raw/master_dataset.csv`
- `data/raw/stereoset_gender.csv`

Notes:

- This uses Hugging Face `datasets` to fetch `LabHC/bias_in_bios` and `stereoset`.
- You are responsible for complying with each dataset’s license and access terms.
- Hugging Face downloads are cached by default (typically under `~/.cache/huggingface/`).

### 4.2 Regenerate processed splits

Processed train/val/test CSVs are produced by:

- `src/data_loader.py`

This script reads `data/raw/master_dataset.csv` and `data/raw/synthetic_career_data.csv`, then writes:

- `data/processed/biasbios_train.csv`
- `data/processed/biasbios_val.csv`
- `data/processed/biasbios_test.csv` (held-out BiasBios-only)

Run:

```bash
python src/data_loader.py
```

Then re-run the Level 1 experiments.

---

## 5. Level 3 — Re-train Baselines (Advanced)

This level re-trains baselines from `data/processed/` and regenerates their artifacts under `models/baselines/`.

### 5.1 TF-IDF + Logistic Regression

```bash
python src/baselines/fit_tfidf.py \
	--processed-dir data/processed \
	--models-dir models/baselines \
	--results-dir results
```

### 5.2 GloVe (or Word2Vec) + SVM

```bash
python src/baselines/glove_svm.py \
	--processed-dir data/processed \
	--models-dir models/baselines \
	--results-dir results
```

Notes:

- This script may download embeddings via `gensim.downloader` (network required).
- Embedding downloads are cached by gensim.

### 5.3 RoBERTa baseline

```bash
python src/baselines/roberta_train.py \
	--processed-dir data/processed \
	--models-dir models/baselines/roberta_baseline \
	--results-dir results
```

### 5.4 CDA baseline

```bash
python src/baselines/debias_cda.py
```

This script writes artifacts under `models/baselines/cda` and outputs predictions/reports to `results/`.

After retraining, re-run:

```bash
python src/benchmark/run_all.py --stage all
```

---

## 6. (Optional) BiasGuard Model Re-training (DistilBERT)

The repository ships a trained BiasGuard checkpoint under `models/`. A fully-packaged “train from scratch” CLI is not currently provided, but the building blocks exist:

- `src/preprocesser.py` (tokenization + DataLoaders)
- `src/model_training.py` (training loop)

Best-effort minimal training invocation (writes to a new directory so you don’t overwrite the shipped checkpoint):

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path

from transformers import AutoTokenizer

from preprocesser import get_dataloaders
from model_training import TrainingConfig, train_model

tokenizer_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
loaders = get_dataloaders(tokenizer_name=tokenizer_name, batch_size=16, max_length=256)

cfg = TrainingConfig(model_name=tokenizer_name, batch_size=16, epochs=3)
out_dir = Path('models/retrained_distilbert')
train_model(loaders, tokenizer, cfg, checkpoint_dir=out_dir / 'checkpoints', final_model_dir=out_dir, report_dir=Path('results'))
PY
```

If you re-train BiasGuard, pass your new checkpoint directory to downstream scripts (e.g., `--model_path models/retrained_distilbert`).

---

## 7. Figures

Figures are generated from `results/metrics_per_model.csv`.

### 7.1 Regenerate figures

```bash
make figures
# or
python scripts/plot_metrics.py
```

Outputs are written to `figures/`.

### 7.2 Regenerate `metrics_per_model.csv`

This file is produced in the notebook:

- `notebooks/Bias_Detection_Model_Evaluation.ipynb`

Run the notebook end-to-end to recompute the summary table and write `results/metrics_per_model.csv`.

---

## 8. Validation / Sanity Checks

Quick checks that the environment and core pipeline are functioning:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

# Unit tests (lightweight)
pytest -q
```

---

## 9. Troubleshooting

- **Import errors (e.g., `No module named preprocesser`):** run scripts as `python src/<script>.py` from repo root, or set `PYTHONPATH=src`.
- **Out-of-memory on GPU:** reduce batch sizes (e.g., `--batch_size 8`) and/or `--max_length`.
- **Slow CPU inference:** expect longer runtimes; prefer running only a subset of stages.
- **Dataset access issues:** some Hugging Face datasets require login/acceptance; follow dataset card instructions.

---

## 10. Licensing and Responsible Use

- Code is licensed under the repository root `LICENSE` (MIT).
- Datasets (BiasBios and StereoSet) have their own licenses; consult dataset cards and follow their citation requirements.
- Outputs are intended for research, auditing, and human-in-the-loop review—not for automated enforcement or high-stakes decision-making.


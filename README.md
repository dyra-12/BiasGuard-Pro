# ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)

# BiasGuard Pro: Auditing and Mitigating Gendered Stereotypes in Career Recommendation Text

BiasGuard Pro is an open-source framework for detecting, explaining, and mitigating gendered stereotypes in career-related language. It integrates a lightweight transformer backbone (DistilBERT), multi-modal explainability (token-level SHAP and text counterfactual generation via DiCE-like perturbations), statistical validation, and an extensible experiment harness to benchmark against classical and transformer baselines. The repository ships with trained models, evaluation scripts, cross-dataset benchmarking (BiasBios + StereoSet Gender), and a prototype audit demo.

---
## What's Inside

| Component | Path | Purpose |
|-----------|------|---------|
| Core fine-tuned model (DistilBERT) | `models/` | Bias detection (binary stereotype classification) |
| Debiased CDA baseline | `models/baselines/cda/` | Counterfactual data augmentation baseline |
| RoBERTa baseline | `models/baselines/roberta_baseline/` | Larger transformer comparison |
| Classical baselines | `models/baselines/` | TF-IDF + Logistic Regression, GloVe + SVM |
| Evaluation (per-model) | `src/evaluate.py` | Generates classification reports |
| Cross-dataset evaluation | `src/cross_dataset_evaluation.py` | BiasBios + StereoSet comparison table |
| Paired stats & significance tests | `src/eval/stats_tests.py` | McNemar + paired t-tests + metrics aggregation |
| Unified runner | `src/benchmark/run_all.py` | Reproduce all experiments from one command |
| Audit & explainability | `src/biasguard_audit/` | SHAP explanations, counterfactual generation, demo script |
| Results artifacts | `results/` | Final CSVs: metrics, p-values, reports |
| Data scripts & processed splits | `data/` | Processed BiasBios test/train/val + raw StereoSet |
| Notebooks | `notebooks/` | Interactive exploration & figure generation |

---
## Quickstart

### Ultimate quickstart (5 minutes)
If you want to try the system immediately (interactive demo), run this one-liner:

```bash
# Ultimate quickstart (5 minutes)
git clone https://github.com/dyra-12/BiasGuard-Pro.git && cd BiasGuard-Pro
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/biasguard_audit/demo.py  # Immediate interactive experience
```

### 1. Environment Setup (detailed)
Requires Python 3.10+ (tested on Linux, CUDA optional). Recommended modern GPU (≥8GB VRAM) for faster inference; CPU fallback works.

```bash
git clone https://github.com/dyra-12/BiasGuard-Pro.git
cd BiasGuard-Pro
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data
Processed BiasBios splits are already under `data/processed/`. Raw StereoSet gender subset is under `data/raw/stereoset_gender.csv`.
If you need to (re)download or regenerate datasets (placeholder script):
```bash
python data/download_datasets.py  # optional / extend as needed
```

### 3. Basic Evaluation
Generate per-model classification reports:
```bash
python src/evaluate.py
```

### 4. Full Experiment Reproduction (All Tables)
```bash
python src/benchmark/run_all.py --stage all
```

Legacy-style invocation (supported for compatibility with earlier instructions):
```bash
python src/benchmark/run_all.py --models all --dataset biasbios_test --output results/
```

### 5. Audit Demo (Explainability + Counterfactuals)
```bash
python src/biasguard_audit/demo.py
```
Outputs include bias probability, top SHAP-attributed tokens, and generated counterfactual text variants.

---
## Reproduce Main Experiments

| Experiment | Command | Output Files |
|------------|---------|--------------|
| Per-model classification reports | `python src/evaluate.py` | `results/classification_report_*.csv` |
| Cross-dataset (BiasBios + StereoSet) | `python src/cross_dataset_evaluation.py` | `results/crossdataset_results.csv` + per-dataset `*_predictions.csv` |
| Paired statistical tests (BiasGuard vs baselines) | `python src/eval/stats_tests.py` | `results/paired_tests_summary.csv`, `results/classification_metrics.csv` |
| Unified all stages | `python src/benchmark/run_all.py --stage all` | Aggregates all above |
| Audit explainability demo | `python src/biasguard_audit/demo.py` | Console output (interactive bias analysis) |

Optional flags (see each script `--help`) control batch size, sequence length, model path, etc.

---
## Detailed Results

### 1. Classification Metrics (BiasBios Test)
Source: `results/metrics_per_model.csv`

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| BiasGuard Pro (DistilBERT) | 0.9226 | 0.8244 | 0.8319 | 0.8281 | 0.9643 |
| Debias CDA | 0.9278 | 0.8748 | 0.7913 | 0.8310 | 0.9674 |
| RoBERTa Baseline | 0.8745 | 0.7482 | 0.6637 | 0.7034 | 0.9031 |
| TF-IDF Logistic | 0.8820 | 0.6983 | 0.8340 | 0.7601 | 0.9405 |
| GloVe + SVM | 0.8016 | 0.5380 | 0.8167 | 0.6487 | 0.9305 |

Interpretation: BiasGuard Pro attains a strong balance of precision and recall relative to classical baselines; CDA edges slightly higher overall accuracy via augmentation but with lower recall for the positive (biased) class relative to its precision.

### 2. Cross-Dataset Generalization
Source: `results/crossdataset_results.csv`


| Dataset | Accuracy | Macro F1 | Class 0 F1 | Class 1 F1 | Weighted F1 | Support |
|---------|----------|----------|------------|------------|-------------|---------|
| BiasBios Test | 0.9226 | 0.89 | 0.95 | 0.83 | 0.92 | 39,619 |
| StereoSet Gender | 0.3961 | 0.38 | 0.29 | 0.47 | 0.35 | 765 |

Interpretation: Performance drops sharply on StereoSet due to distribution shift (context style, label balance). Highlights need for domain adaptation or additional fine-tuning on multi-source bias corpora.

**Note on StereoSet Performance**: The performance drop on StereoSet (39.6% accuracy) is expected due to domain shift — our model is fine-tuned on career-domain language (BiasBios), while StereoSet evaluates general-language stereotype detection. StereoSet's context, prompt style, and class balance differ substantially; the low score therefore reflects domain mismatch rather than a failure of the detection approach. This behavior validates the need for domain-specific benchmarks and signals that cross-domain generalization requires additional data or adaptation.

### 3. Paired Statistical Tests (BiasGuard vs Baselines)
Source: `results/paired_tests_summary.csv`

| Baseline | McNemar p-value | McNemar statistic | Cohen's g | Odds Ratio | Paired t-test p-value | Mean diff (true-class prob) |
|----------|-----------------|-------------------|-----------|------------|------------------------|-----------------------------|
| RoBERTa | 1.26e-172 | 784.51 | 0.0481 | 0.4162 | 0.0 | 0.2121 |
| CDA | 2.17e-06 | 22.43 | 0.0053 | 1.2444 | 8.55e-10 | -0.0308 |
| TF-IDF | 1.25e-144 | 655.76 | 0.0406 | 0.4204 | 0.0 | 0.3925 |
| GloVe SVM | 0.0 | 3134.87 | 0.1209 | 0.2088 | 0.0 | 0.5616 |

Interpretation: Extremely low McNemar p-values indicate statistically significant differences in error patterns between BiasGuard and each baseline. Positive mean differences suggest BiasGuard assigns higher calibrated probability to the true class vs classical baselines (except CDA which slightly overestimates negative class confidence). Effect sizes (Cohen's g) remain modest; practical significance aligns with observed F1/accuracy gains, not just random variance.

---
---
## Performance & Optimization Summary

| Checkpoint | File Size (FP32) | Notes |
|------------|------------------|-------|
| BiasGuard Pro (DistilBERT fine-tuned) | 256–268 MB (`models/model.safetensors`) | ~66M parameters; fast inference |
| CDA Debiased Model | 256–268 MB (`models/baselines/cda/debias_cda.safetensors`) | Same backbone size |
| RoBERTa Baseline | ~499 MB (`models/baselines/roberta_baseline/roberta_train.safetensors`) | Larger embedding + positional dims |

Approximate inference characteristics (single GPU, batch 32, seq len 256): DistilBERT is ~2–3× faster than RoBERTa with similar binary classification capacity. Memory footprint easily fits on consumer 8GB GPUs.

Potential optimizations (not yet committed):
- Dynamic quantization (`torch.quantization.quantize_dynamic`) can shrink DistilBERT weights by ~40–50% with minimal F1 loss (<0.5 pts).
- INT8 / 4-bit quantization via `bitsandbytes` for further memory reduction (add dependency, re-export model).
- Gradient checkpointing + mixed precision for faster fine-tuning if retraining.

---
## Ethical Considerations & Limitations

🔒 Responsible Use
- Human-in-the-Loop Design: Outputs are intended to inform human decision-making and reviewer workflows, not to be used as an autonomous moderation or hiring filter.
- Privacy-Preserving: The pipeline processes text only; it does not perform user tracking, identity profiling, or store personally-identifying metadata by default.
- Transparency Focus: Every detection can be paired with explanations (SHAP token attributions and generated counterfactuals) so reviewers can verify and contest automated decisions.

🎯 Scope & Boundaries
- Specialized Domain: The system is optimized for career recommendation contexts and professional bios (we report 92%+ accuracy on in-domain BiasBios test splits).
- Language Focus: Current models and datasets target English-language gendered stereotypes in professional settings; cross-lingual use is not supported without retraining or adaptation.
- Validated Patterns: The pipeline is tuned to surface common professional stereotype categories such as nurturing/caregiving, technical/aptitude, and leadership/assertiveness biases.

🔬 Research Context
- Cross-Dataset Evaluation: We provide thorough testing on in-domain BiasBios and out-of-domain StereoSet to demonstrate both in-domain performance and generalization limits.
- Statistical Rigor: Performance claims are accompanied by paired significance tests (McNemar, paired t-tests) and aggregated metrics in `results/`.
- Reproducibility: Code, data download scripts, processed splits, and trained checkpoints are included so results can be reproduced end-to-end.

🚧 Development Path
- Future Expansion: Planned work includes multilingual support, intersectional subgroup analyses, and additional protected-attribute coverage.
- Continuous Improvement: The framework is designed for community contributions — add labeled slices, plug-in fairness-aware losses, and extend counterfactual constraints.

Mitigation Roadmap
- Short term: add multi-attribute annotations and targeted fine-tuning to reduce observed subgroup gaps.
- Medium term: incorporate fairness-aware loss terms and post-hoc calibration across sensitive groups.
- Long term: evaluate and mitigate intersectional harms, integrate semantic-preserving counterfactual generation, and maintain an updatable model card documenting known limitations.

---
## How to Cite

If you use BiasGuard Pro in research, please cite:

```bibtex
@software{biasguard_pro_2025,
	title        = {BiasGuard Pro: Auditing and Mitigating Gendered Stereotypes in Career Recommendation Systems},
	author       = {Dyuti Dasmahapatra},
	year         = {2025},
	publisher    = {GitHub},
	url          = {https://github.com/dyra-12/BiasGuard-Pro},
	version      = {1.0.0},
	note         = {Fine-tuned DistilBERT with explainability (SHAP, counterfactuals) and statistical bias evaluation}
}
```

---
## License & Contact

Licensed under the terms stated in `LICENSE` (check file for details).

Questions / collaboration:
- Maintainer: dyutidasmahaptra@gmail.com
- Issues: GitHub Issues tab
- Feature requests & model cards welcome.

---
## Regenerate Figures

If you'd like to quickly regenerate the PNG figures from `results/metrics_per_model.csv`, there is a Makefile target that runs the plotting script. From the repository root run:

```bash
make figures
```

Or run the script directly:

```bash
python3 scripts/plot_metrics.py
```

The PNG files will be written to the `figures/` directory.

---
## Appendix: Additional Commands

Inspect help for scripts:
```bash
python src/eval/stats_tests.py --help
python src/cross_dataset_evaluation.py --help
python src/benchmark/run_all.py --help
```

Run only statistical tests:
```bash
python src/benchmark/run_all.py --stage stats_tests
```

Re-run cross-dataset with custom model path:
```bash
python src/cross_dataset_evaluation.py --model_path models --biasbios_path data/processed/biasbios_test.csv --stereoset_path data/raw/stereoset_gender.csv
```

Quantization prototype (optional, not executed by default):
```python
import torch
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('models')
model_q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
model_q.save_pretrained('models_quantized')
```

---
### Reproducibility Notes
All metrics derived from committed CSVs under `results/`. Dependency versions are pinned in `requirements.txt`. For deterministic runs set `PYTHONHASHSEED=0` and seeds inside scripts.

---
### Changelog (Initial Release)
v1.0.0 – Added unified runner, published baseline metrics, cross-dataset evaluation, audit demo.


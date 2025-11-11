Figures directory — organized

This folder contains all generated figures for the BiasGuard-Pro project. I reorganized the images into a consistent layout and renamed them to make their purpose and provenance clear.

Top-level layout

- models/
  - baseline/
  - cda/
  - glove_svm/
  - roberta/
  - tfidf_logreg/

  Each model folder contains three canonical plots:
  - roc_curve.png — ROC curve (receiver operating characteristic)
  - pr_curve.png — Precision-Recall curve
  - confusion_matrix.png — Confusion matrix for the main test split

- summary/
  Summary and cross-model comparison plots. Files include:
  - accuracy_by_model.png — accuracy scores by model
  - metrics_grouped_bar.png — grouped metrics (precision/recall/f1) per model
  - metrics_radar.png — radar/spider chart comparing models on selected metrics
  - model_performance_comparison.png — overall performance comparison (formerly `model_performance_comparision.png`)
  - precision_recall_f1_bar.png — bar chart showing precision/recall/f1 side-by-side (formerly `prf_bar.png`)
  - roc_auc_by_model.png — ROC AUC values by model (formerly `roc_auc_bar.png`)

Why these names and layout

- Grouping by model makes it easy to find all plots produced for a single model when inspecting behavior, errors, and trade-offs.
- Standardizing filenames (e.g. `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`) allows scripts and notebooks to reference images predictably.
- Summary plots live in `summary/` to emphasize that they compare models rather than describing a single model.

Mapping from previous filenames to new locations

- `accuracy_bar.png` -> `summary/accuracy_by_model.png`
- `base_confusion_matrix.png` -> `models/baseline/confusion_matrix.png`
- `base_pr.png` -> `models/baseline/pr_curve.png`
- `base_roc.png` -> `models/baseline/roc_curve.png`
- `cda_confusion_matrix.png` -> `models/cda/confusion_matrix.png`
- `cda_pr.png` -> `models/cda/pr_curve.png`
- `cda_roc.png` -> `models/cda/roc_curve.png`
- `glove_svm_confusion_matrix.png` -> `models/glove_svm/confusion_matrix.png`
- `glove_svm_pr.png` -> `models/glove_svm/pr_curve.png`
- `glove_svm_roc.png` -> `models/glove_svm/roc_curve.png`
- `metrics_grouped_bar.png` -> `summary/metrics_grouped_bar.png`
- `metrics_radar.png` -> `summary/metrics_radar.png`
- `model_performance_comparision.png` -> `summary/model_performance_comparison.png` (typo fixed)
- `prf_bar.png` -> `summary/precision_recall_f1_bar.png` (renamed for clarity)
- `roberta_confusion_matrix.png` -> `models/roberta/confusion_matrix.png`
- `roberta_pr.png` -> `models/roberta/pr_curve.png`
- `roberta_roc.png` -> `models/roberta/roc_curve.png`
- `roc_auc_bar.png` -> `summary/roc_auc_by_model.png`
- `tfidf_logreg_confusion_matrix.png` -> `models/tfidf_logreg/confusion_matrix.png`
- `tfidf_logreg_pr.png` -> `models/tfidf_logreg/pr_curve.png`
- `tfidf_logreg_roc.png` -> `models/tfidf_logreg/roc_curve.png`

Descriptions (short) — per-model

- models/<model>/roc_curve.png
  ROC curve plotting true positive rate vs false positive rate. Helpful to compare AUC performance and threshold behavior.

- models/<model>/pr_curve.png
  Precision-Recall curve useful for imbalanced data and understanding precision/recall trade-offs.

- models/<model>/confusion_matrix.png
  Confusion matrix for the main test set — quick visual of false positives vs false negatives.

Descriptions (short) — summary

- summary/accuracy_by_model.png
  Bar chart of accuracy for each model on the selected test set.

- summary/metrics_grouped_bar.png
  Grouped bars showing precision, recall, and F1 per model for side-by-side comparison.

- summary/metrics_radar.png
  Radar chart showing multiple metric axes (e.g., precision, recall, f1, auc) for each model.

- summary/model_performance_comparison.png
  A compact comparison of overall model performance (composite view used in the paper/README).

- summary/precision_recall_f1_bar.png
  Alternate bar chart layout for precision/recall/F1 for each model.

- summary/roc_auc_by_model.png
  Shows the AUC-ROC values per model.

Regenerating figures

Most figures are produced by the evaluation and plotting utilities in this repository. Typical places to look:
- `scripts/plot_metrics.py` — plotting utilities (check this script to re-run summary plots)
- `src/` — training and evaluation code (e.g. `evaluate.py`, `cross_dataset_evaluation.py`) which produce per-model metrics and raw outputs used to build the above plots
- Notebooks such as `notebooks/Bias_Detection_Model_Evaluation.ipynb` — interactive exploration and plotting

A suggested flow to regenerate everything
1. Re-run evaluations / predictions to produce the metrics and prediction CSVs (see `src/` and `scripts/`) — these will write the metric CSVs used by plotting scripts.
2. Run `python scripts/plot_metrics.py` (or the relevant notebook) to regenerate the plots into this `figures/` folder. Adjust script paths if the plotting script writes to a different path.

Notes, assumptions & git

- I assumed the objective was to group plots by model and place cross-model comparisons in `summary/`. If you'd prefer grouping by plot type (i.e., `roc/`, `pr/`, `confusion/`) instead, tell me and I can reorganize.
- Original images were moved and renamed in the working tree. If you use git, the rename/move will show as deletes/adds unless you commit; git tracks content and may show renames automatically. To review changes run `git status` and `git diff --name-status`.

If you'd like:
- I can revert to the prior flat layout or apply a different naming scheme.
- I can add small thumbnails or an index HTML to preview the figures.
- I can update notebooks/scripts to save plots directly to this folder with the canonical names.

If you want any of those follow-ups, tell me which and I'll implement them next.
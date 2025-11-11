#!/usr/bin/env python3
"""Generate performance charts from results/metrics_per_model.csv.

Produces the following PNG files in the `figures/` directory:
 - metrics_grouped_bar.png    (grouped bar chart of Accuracy/Precision/Recall/F1/ROC-AUC)
 - prf_bar.png                (bar chart for Precision/Recall/F1)
 - accuracy_bar.png           (bar chart for Accuracy)
 - roc_auc_bar.png            (bar chart for ROC-AUC)
 - metrics_radar.png          (radar chart of Precision/Recall/F1 for each model)

Charts are saved at high resolution and include clear titles, axis labels,
legends, and grid lines for publication-quality visuals.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Files and directories
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "metrics_per_model.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    raise SystemExit(f"Required CSV not found: {CSV_PATH}")

# Read data
df = pd.read_csv(CSV_PATH)
# Ensure model names are readable labels
df['model_label'] = df['model'].str.replace('_', ' ').str.title()
df = df.set_index('model_label')
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Basic styling: prefer seaborn-whitegrid when available, otherwise fall back to ggplot
if 'seaborn-whitegrid' in plt.style.available:
    plt.style.use('seaborn-whitegrid')
else:
    plt.style.use('ggplot')

# 1) Grouped bar chart for the main metrics
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.12
x = np.arange(len(df.index))
for i, m in enumerate(metrics):
    ax.bar(x + (i - 2) * width, df[m], width=width, label=m.replace('_', ' ').title())

ax.set_xticks(x)
ax.set_xticklabels(df.index, rotation=45, ha='right')
ax.set_ylim(0, 1.02)
ax.set_ylabel('Score (0-1)', fontsize=12)
ax.set_title('Model Performance Comparison — Key Metrics', fontsize=14, weight='bold')
ax.legend(title='Metric')
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
out = FIG_DIR / 'metrics_grouped_bar.png'
fig.savefig(out, dpi=300)
plt.close(fig)

# 2) Precision/Recall/F1 grouped
fig, ax = plt.subplots(figsize=(10, 5))
prf = ['precision', 'recall', 'f1']
width = 0.2
x = np.arange(len(df.index))
for i, m in enumerate(prf):
    ax.bar(x + (i - 1) * width, df[m], width=width, label=m.title())
ax.set_xticks(x)
ax.set_xticklabels(df.index, rotation=45, ha='right')
ax.set_ylim(0, 1.02)
ax.set_ylabel('Score (0-1)')
ax.set_title('Precision / Recall / F1 — Per Model')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
out = FIG_DIR / 'prf_bar.png'
fig.savefig(out, dpi=300)
plt.close(fig)

# 3) Accuracy bar
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(df.index, df['accuracy'], color='tab:blue')
ax.set_ylim(0, 1.02)
ax.set_ylabel('Accuracy (0-1)')
ax.set_title('Accuracy per Model')
ax.set_xticklabels(df.index, rotation=45, ha='right')
for i, v in enumerate(df['accuracy']):
    ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
out = FIG_DIR / 'accuracy_bar.png'
fig.savefig(out, dpi=300)
plt.close(fig)

# 4) ROC-AUC bar
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(df.index, df['roc_auc'], color='tab:green')
ax.set_ylim(0, 1.02)
ax.set_ylabel('ROC AUC (0-1)')
ax.set_title('ROC-AUC per Model')
ax.set_xticklabels(df.index, rotation=45, ha='right')
for i, v in enumerate(df['roc_auc']):
    ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
out = FIG_DIR / 'roc_auc_bar.png'
fig.savefig(out, dpi=300)
plt.close(fig)

# 5) Radar chart for Precision / Recall / F1
# Helper to create radar
categories = prf
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

for idx, row in df.iterrows():
    values = row[prf].tolist()
    values += values[:1]
    ax.plot(angles, values, label=idx, linewidth=2)
    ax.fill(angles, values, alpha=0.15)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), [c.title() for c in categories])
ax.set_ylim(0, 1)
ax.set_title('PRF Radar — Precision, Recall, F1', y=1.08)
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
out = FIG_DIR / 'metrics_radar.png'
fig.savefig(out, dpi=300)
plt.close(fig)

print(f"Saved charts to {FIG_DIR}")

#!/usr/bin/env python3
"""Counterfactual Data Augmentation (CDA) + RoBERTa fine-tune baseline.

This script expects the processed CSV files at data/processed/biasbios_train.csv,
data/processed/biasbios_val.csv, data/processed/biasbios_test.csv.

It will:
- Load and sanitize splits
- Apply simple gender-word swaps to augment the training split (CDA)
- Fine-tune a Hugging Face Transformer classifier on augmented data
- Save the model & tokenizer under models/baselines/cda
- Write test predictions to results/predictions_cda.csv and a tidy
  classification report CSV to results/classification_report_cda.csv

Note: This script is written for static review and ready-to-run in a proper
Python environment (matching numpy/pandas/torch builds). Do not run inside the
editor environment if binary compatibility errors are present.
"""

from __future__ import annotations

import os
import json
import re
import time
from typing import Dict, List, Tuple

os.environ.setdefault("WANDB_DISABLED", "true")

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

from datasets import Dataset
from transformers import (
	AutoTokenizer,
	AutoModelForSequenceClassification,
	TrainingArguments,
	Trainer,
	DataCollatorWithPadding,
)


# Config / paths
MODEL_ID = "roberta-base"
OUT_DIR = "models/baselines/cda"
RESULTS_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

PRED_CSV = os.path.join(RESULTS_DIR, "predictions_cda.csv")
REPORT_CSV = os.path.join(RESULTS_DIR, "classification_report_cda.csv")
LABEL_MAP = os.path.join(OUT_DIR, "label_mapping.json")

# Training hyperparams (conservative defaults)
NUM_EPOCHS = 3
TRAIN_BSZ = 16
EVAL_BSZ = 32
LR = 2e-5
MAX_LEN = 128


def _detect_text_label_cols(df: pd.DataFrame) -> Tuple[str, str]:
	"""Try to infer the text and label column names used in processed CSVs."""
	candidates_text = [c for c in df.columns if c.lower() in ("text", "sentence", "excerpt", "bio")]
	candidates_label = [c for c in df.columns if c.lower() in ("label", "label_id", "target", "profession", "class")] + ["gold_label"]
	text_col = candidates_text[0] if candidates_text else "text"
	label_col = None
	for c in candidates_label:
		if c in df.columns:
			label_col = c
			break
	if label_col is None:
		# fallback: any column with few unique values
		for c in df.columns:
			if df[c].nunique() < 50 and c != text_col:
				label_col = c
				break
	if label_col is None:
		raise ValueError("Could not infer label column from processed CSV. Please check headers.")
	return text_col, label_col


def load_processed_splits(base_path: str = "data/processed"):
	train = pd.read_csv(os.path.join(base_path, "biasbios_train.csv"))
	val = pd.read_csv(os.path.join(base_path, "biasbios_val.csv"))
	test = pd.read_csv(os.path.join(base_path, "biasbios_test.csv"))

	tx, ly = _detect_text_label_cols(train)
	# Ensure consistent columns
	train = train[[tx, ly]].rename(columns={tx: "text", ly: "label"})
	val = val[[tx, ly]].rename(columns={tx: "text", ly: "label"}) if tx in val.columns and ly in val.columns else val
	if "text" not in val.columns or "label" not in val.columns:
		# try detect independently for val/test
		t2, l2 = _detect_text_label_cols(val)
		val = val[[t2, l2]].rename(columns={t2: "text", l2: "label"})
	t3, l3 = _detect_text_label_cols(test)
	test = test[[t3, l3]].rename(columns={t3: "text", l3: "label"})

	return train, val, test


def sanitize_text(x) -> str:
	if x is None:
		return ""
	if isinstance(x, float) and np.isnan(x):
		return ""
	return str(x).strip()


def build_gender_lexicon() -> Dict[str, str]:
	pairs = [
		("he", "she"), ("him", "her"), ("his", "hers"), ("himself", "herself"),
		("man", "woman"), ("men", "women"), ("boy", "girl"), ("male", "female"),
		("father", "mother"), ("son", "daughter"), ("brother", "sister"), ("husband", "wife"),
		("actor", "actress"), ("waiter", "waitress"), ("mr", "ms"),
	]
	lex = {}
	for a, b in pairs:
		lex[a] = b
		lex[b] = a
	# 'her' will map to 'him' or 'his' depending on simple heuristic when used
	lex["hers"] = "his"
	lex["his"] = "hers"
	return lex


def simple_token_words(text: str) -> List[str]:
	return re.findall(r"\w+|\s+|[^\w\s]", text, flags=re.UNICODE)


def swap_gender_words(text: str, lex: Dict[str, str]) -> str:
	tokens = simple_token_words(text)
	out = []
	i = 0
	while i < len(tokens):
		tok = tokens[i]
		if re.match(r"^\w+$", tok):
			low = tok.lower()
			if low == "her":
				# heuristics: if next token is a word (likely determiner) -> his else him
				j = i + 1
				while j < len(tokens) and tokens[j].isspace():
					j += 1
				repl = "his" if j < len(tokens) and re.match(r"^\w+$", tokens[j]) else "him"
				if tok.istitle():
					repl = repl.title()
				elif tok.isupper():
					repl = repl.upper()
				out.append(repl)
			elif low in lex:
				repl = lex[low]
				if tok.istitle():
					repl = repl.title()
				elif tok.isupper():
					repl = repl.upper()
				out.append(repl)
			else:
				out.append(tok)
		else:
			out.append(tok)
		i += 1
	return "".join(out)


def cda_augment(train_texts: List[str], train_labels: List[int], only_if_present: bool = True) -> Tuple[List[str], List[int]]:
	lex = build_gender_lexicon()
	X_aug: List[str] = []
	y_aug: List[int] = []
	for t, l in zip(train_texts, train_labels):
		t_clean = sanitize_text(t)
		swapped = swap_gender_words(t_clean, lex)
		if only_if_present and swapped == t_clean:
			X_aug.append(t_clean)
			y_aug.append(l)
		else:
			# keep original and swapped
			X_aug.append(t_clean)
			y_aug.append(l)
			X_aug.append(swapped)
			y_aug.append(l)
	return X_aug, y_aug


def save_classification_report_csv(report_dict: Dict, out_path: str):
	# report_dict as produced by sklearn.classification_report(..., output_dict=True)
	rows = []
	for label, metrics in report_dict.items():
		if label in ("accuracy", "macro avg", "weighted avg"):
			rows.append({"label": label, **{k: metrics[k] if isinstance(metrics, dict) and k in metrics else (metrics if k == "precision" else None) for k in ("precision","recall","f1-score","support")}})
		else:
			rows.append({"label": label, "precision": metrics.get("precision"), "recall": metrics.get("recall"), "f1-score": metrics.get("f1-score"), "support": metrics.get("support")})
	df = pd.DataFrame(rows)
	df.to_csv(out_path, index=False)


def build_tokenized_datasets(tokenizer, X_train, y_train, X_val, y_val, X_test, y_test):
	train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
	val_ds = Dataset.from_dict({"text": X_val, "label": y_val})
	test_ds = Dataset.from_dict({"text": X_test, "label": y_test})

	def tok_fn(examples):
		texts = [sanitize_text(t) for t in examples["text"]]
		return tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN)

	tok_train = train_ds.map(tok_fn, batched=True)
	tok_val = val_ds.map(tok_fn, batched=True)
	tok_test = test_ds.map(tok_fn, batched=True)

	cols = ["input_ids", "attention_mask", "label"]
	tok_train.set_format(type="torch", columns=cols)
	tok_val.set_format(type="torch", columns=cols)
	tok_test.set_format(type="torch", columns=cols)
	return tok_train, tok_val, tok_test


def main():
	train, val, test = load_processed_splits()
	# sanitize and drop empty
	train = train.dropna(subset=["text", "label"]).copy()
	val = val.dropna(subset=["text", "label"]).copy()
	test = test.dropna(subset=["text", "label"]).copy()

	# label encoding
	le = LabelEncoder()
	y_train = le.fit_transform(train["label"].astype(str).tolist())
	y_val = le.transform(val["label"].astype(str).tolist())
	y_test = le.transform(test["label"].astype(str).tolist())

	X_train_raw = train["text"].astype(str).tolist()
	X_val_raw = val["text"].astype(str).tolist()
	X_test_raw = test["text"].astype(str).tolist()

	# CDA augmentation on training split
	X_train_aug, y_train_aug = cda_augment(X_train_raw, list(y_train), only_if_present=True)

	# persist label mapping
	mapping = {int(i): str(c) for i, c in enumerate(le.classes_)}
	with open(LABEL_MAP, "w", encoding="utf-8") as f:
		json.dump(mapping, f, indent=2)

	# tokenizer & model
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
	model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=len(le.classes_))

	tok_train, tok_val, tok_test = build_tokenized_datasets(tokenizer, X_train_aug, y_train_aug, X_val_raw, list(y_val), X_test_raw, list(y_test))

	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

	args = TrainingArguments(
		output_dir=OUT_DIR,
		num_train_epochs=NUM_EPOCHS,
		per_device_train_batch_size=TRAIN_BSZ,
		per_device_eval_batch_size=EVAL_BSZ,
		learning_rate=LR,
		evaluation_strategy="epoch",
		save_strategy="epoch",
		load_best_model_at_end=True,
		metric_for_best_model="eval_loss",
		logging_dir=os.path.join(OUT_DIR, "logs"),
		report_to=None,
	)

	def compute_metrics(p):
		preds = np.argmax(p.predictions, axis=1)
		precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average="weighted", zero_division=0)
		acc = accuracy_score(p.label_ids, preds)
		return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

	trainer = Trainer(
		model=model,
		args=args,
		train_dataset=tok_train,
		eval_dataset=tok_val,
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics,
	)

	trainer.train()
	trainer.save_model(OUT_DIR)
	tokenizer.save_pretrained(OUT_DIR)

	# (Evaluation removed; handled centrally in unified evaluate.py)
	print("CDA training complete. Model artifacts saved to:")
	print(f" - model dir: {OUT_DIR}")


if __name__ == "__main__":
	main()

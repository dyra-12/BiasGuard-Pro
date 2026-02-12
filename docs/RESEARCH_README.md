# ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)


# BiasGuard Pro

**Human-Centered Fairness Auditing for Career Recommendation Systems**

A research framework for auditing and mitigating gendered stereotypes in professional language through interpretable, human-in-the-loop reasoning.

---

## Abstract

Career recommendation systems increasingly influence how individuals explore and pursue professional opportunities. While powerful, these systems often reproduce subtle gendered stereotypes embedded in training data and linguistic conventions. Existing bias detection approaches either prioritize benchmark accuracy without interpretability or provide opaque industry APIs with limited support for human judgment.

This repository presents **BiasGuard Pro**, a human-centered framework for auditing gendered bias in career recommendation text. The system combines a lightweight transformer-based bias detector with multi-modal explanations—including token-level attribution and counterfactual rewrites—designed explicitly to support human reasoning under uncertainty.

BiasGuard Pro reframes fairness auditing as an **interpretive process**, not a fully automated decision. Empirical evaluation demonstrates strong in-domain performance, reduced gender disparity, statistically significant improvements over baselines, and stable, human-interpretable explanations. The framework is intended as a deployable research contribution at the intersection of fairness-aware NLP, explainable AI, and human-centered AI.

---

## How to Read This Repository

Start here for a conceptual overview, then explore:

- **`docs/ARCHITECTURE.md`** — System design and human–AI interaction rationale
- **`docs/DATA.md`** — Dataset construction, ethics, and limitations
- **`docs/RESULTS.md`** — Performance, fairness, and interpretability findings
- **`docs/EXPLAINABILITY.md`** — Explanation design as cognitive support
- **`docs/INTERFACE.md`** — Interaction design and trust calibration
- **`docs/ETHICS.md`** — Responsible use, scope, and ethical positioning
- **`docs/REPRODUIBILITY.md`** — Guide for Reproducibility 

---

## Core Idea

> Most fairness systems optimize **what to detect**.  
> BiasGuard Pro focuses on **how humans reason** about what is detected.

At each audit step, the system:

1. Estimates bias likelihood in career-related text
2. Surfaces localized linguistic attributions supporting causal reasoning
3. Generates counterfactual alternatives enabling "what-if" exploration
4. Maintains human authority over interpretation and action

This design makes fairness auditing:

- **Inspectable**
- **Context-sensitive**
- **Resistant to automation bias**
- **Grounded in human judgment**

---

## System Architecture

BiasGuard Pro is an interaction-centered auditing system composed of:

### 1. Bias Detection Engine

A DistilBERT-based classifier optimized for interpretability, efficiency, and real-time use.

### 2. Attribution Module

Token-level SHAP explanations highlighting linguistic contributors to bias predictions.

### 3. Counterfactual Generator

Minimal, semantically coherent rewrites illustrating how bias signals change under linguistic perturbation.

### 4. Interaction Layer

Progressive disclosure and uncertainty signaling to support calibrated trust and reflective reasoning.

**Key properties:**

- CPU-only inference (~12 ms latency)
- No automated enforcement
- Explanations treated as evidence, not verdicts

See **`docs/ARCHITECTURE.md`** for full system reasoning.

---

## Evaluation and Findings

BiasGuard Pro is evaluated across four axes:

### Predictive Performance

- **In-domain F1**: 0.828
- **ROC–AUC**: 0.964
- Outperforms classical and transformer baselines

### Fairness Robustness

- **Disparity Gap**: 0.041
- **Bias Amplification**: 1.06
- Reduced gender asymmetry without sacrificing accuracy

### Statistical Validation

- **McNemar tests**: p < 0.001 across baselines
- Improvements are statistically significant, not incidental

### Human-Centered Interpretability

- Stable SHAP explanations (cosine similarity ≈ 0.91)
- Formative user evaluation (n = 6) shows improved causal articulation and trust calibration

Detailed analysis is provided in **`docs/RESULTS.md`**.

---

## Human-Centered Explainability

BiasGuard Pro integrates two complementary explanation modalities:

### Attribution (Why?)

Token-level SHAP values support attributional reasoning under ambiguity.

### Counterfactuals (How?)

Minimal rewrites support mental simulation and bias mitigation reasoning.

Explanations are revealed through **progressive disclosure**, preventing cognitive overload and discouraging blind reliance.

See **`docs/EXPLAINABILITY.md`**.

---

## Ethics, Responsibility, and Scope

This repository presents **non-clinical, minimal-risk research**.

- BiasGuard Pro is an auditing and educational tool, **not a decision-maker**
- Human oversight is required by design
- No personal data is collected or inferred
- Known limitations (binary gender, English-only, domain specificity) are explicitly documented

See **`docs/ETHICS.md`** for full discussion.

---

## Interactive Demo

A hosted demo reproducing the human-centered auditing interface is available:

**Hugging Face Space:**  
https://huggingface.co/spaces/Dyra1204/BiasGuard-Pro

---

## Repository Structure

```
├── README.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA.md
│   ├── RESULTS.md
│   ├── EXPLAINABILITY.md
│   ├── INTERFACE.md
│   └── ETHICS.md
│   └── REPRODUIBILITY.md
├── src/
├── notebooks/
├── assets/
├── LICENSE
├── requirements.txt
└── CITATION.cff
```

---

## Intended Use

### Intended for:

- Fairness auditing in career recommendation systems
- Research in human-centered and explainable AI
- Bias-aware NLP analysis and education

### Not intended for:

- Automated moderation or enforcement
- Penalization or ranking of individuals
- Deployment without human review

---

## Citation

If you use or build on this work, please cite it as specified in **`CITATION.cff`**.

---

## Summary

BiasGuard Pro demonstrates that **fairness in language systems is not achieved through automation alone**. By designing bias detection, explanation, and interaction as a single human-centered system, the framework shows how interpretability can support reflective, accountable fairness auditing in real-world professional contexts.

Rather than treating bias as a binary outcome, BiasGuard Pro reframes it as an **interactive reasoning process** between humans and AI systems.

---

**Last Updated**: January 2026  
**Repository Status**: Active research | System documentation complete | Preparing for dissemination

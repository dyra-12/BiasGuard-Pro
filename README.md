
# BiasGuard Pro: Auditing and Mitigating Gendered Stereotypes in Career Recommendation Texts

## Overview

BiasGuard Pro is a research framework for identifying and auditing gendered stereotypes in professional and career recommendation text. The system is designed to support human interpretation and judgment in fairness-sensitive contexts, rather than to provide automated enforcement or decision-making.

The framework integrates a lightweight bias detection model with interpretable explanations and interaction-level design principles drawn from Human-Centered AI (HCAI) and Explainable AI (XAI).

---

## Motivation

Career recommendation systems increasingly mediate access to professional opportunities. While effective at personalization, these systems can reproduce subtle gendered stereotypes embedded in training data and linguistic conventions.

Existing bias detection tools often prioritize benchmark performance while offering limited support for:

- Interpretability
- Trust calibration
- Human-in-the-loop reasoning

BiasGuard Pro addresses this gap by reframing bias auditing as an **interpretive, human-centered process** rather than a fully automated classification task.

---

## Conceptual Contribution

Most bias detection systems focus on determining **whether** content is biased. BiasGuard Pro emphasizes **how humans reason** about bias signals once they are surfaced.

For each input, the system:

1. Estimates the likelihood of gendered bias
2. Identifies influential linguistic features
3. Generates counterfactual alternatives illustrating potential mitigation
4. Preserves human authority over interpretation and action

This design supports **reflective analysis** rather than categorical judgment.

---

## System Components

BiasGuard Pro consists of:

- A lightweight **transformer-based bias detector** optimized for efficiency and interpretability
- **Token-level attribution** to support causal reasoning about language
- **Counterfactual explanations** enabling "what-if" analysis
- An **interaction model** based on progressive disclosure to reduce cognitive overload and automation bias
- A research-grade **evaluation pipeline** spanning performance, fairness, and interpretability

---

## Empirical Snapshot

- **In-domain F1 score**: 0.828
- **Disparity Gap**: 0.041
- **Bias Amplification**: 1.06
- **Average inference latency**: ~12 ms (CPU-only)

Results indicate that interpretability and fairness robustness can be achieved without sacrificing efficiency.

---

## Interactive Demonstration

An interactive demonstration of the auditing interface is available:

🔗 **Hugging Face Space**  
https://huggingface.co/spaces/Dyra1204/BiasXplainer

---

## Go Deeper

Readers seeking full technical, methodological, and ethical detail may consult:

- **Research overview**: `docs/RESEARCH_README.md`
- **System architecture**: `docs/ARCHITECTURE.md`
- **Data construction and limitations**: `docs/DATA.md`
- **Empirical evaluation**: `docs/RESULTS.md`
- **Explainability design**: `docs/EXPLAINABILITY.md`
- **Interaction rationale**: `docs/INTERFACE.md`
- **Ethical considerations**: `docs/ETHICS.md`

---

## Intended Scope

### Intended for:

- Fairness-aware NLP research
- Human-centered AI studies
- Bias auditing in professional language systems

### Not intended for:

- Automated moderation or enforcement
- High-stakes decision-making without human oversight
- Direct deployment as a standalone decision system

---

## Citation

Citation details are provided in `CITATION.cff`.

---

**Project Status**: Active research  
**Primary Domain**: Career recommendation and professional discourse  
**Design Orientation**: Human-centered, interpretability-first fairness auditing
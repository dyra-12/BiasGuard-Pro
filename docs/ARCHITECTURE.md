# System Architecture: BiasGuard Pro

## 1. Architectural Philosophy

BiasGuard Pro is designed as a **human–AI interaction system**, not a standalone machine learning model.

The architecture reflects a core assumption:

> Fairness auditing is a socio-technical process that requires both algorithmic rigor and human reasoning support.

Accordingly, the system architecture prioritizes:

- Interpretability over raw model complexity
- Modular design for auditability and extension
- Real-time responsiveness for interactive use
- Explicit integration points for human oversight

Rather than optimizing for end-to-end automation, the system is architected to keep humans in the decision loop at every critical stage.

---

## 2. High-Level System Overview

BiasGuard Pro follows a five-stage pipeline, with each stage designed to support a specific human or technical requirement:

1. **Data Curation & Preparation**
2. **Bias Detection Engine**
3. **Explainability Engine**
4. **Interaction Layer**
5. **Evaluation & Feedback Loop**

These stages are loosely coupled but semantically aligned, enabling independent improvement without disrupting the overall system.

---

## 3. Data Layer: Controlled Realism

### 3.1 Design Rationale

Bias detection systems face a tension between:

- **Ecological validity** (real-world language)
- **Controlled attribution** (knowing why bias is detected)

BiasGuard Pro resolves this by combining:

- Large-scale real professional text (BiasBios)
- Small, paired synthetic examples for bias isolation

This design ensures the model learns from authentic language while retaining a stable interpretive anchor for explanation analysis.

### 3.2 Architectural Implication

Synthetic data is treated as a **structural component**, not as a volume driver.

Its role is to:

- Anchor interpretability
- Reduce spurious correlations
- Enable counterfactual reasoning downstream

---

## 4. Bias Detection Engine

### 4.1 Model Selection Rationale

The detection engine is built on **DistilBERT**, chosen deliberately over larger transformer models.

This choice reflects three architectural priorities:

| Criterion | Architectural Impact |
|-----------|---------------------|
| Interpretability | Enables feasible token-level attribution |
| Efficiency | Supports CPU-only, real-time inference |
| Stability | Reduces overfitting to lexical shortcuts |

The system favors **sufficient expressiveness** over maximal representational power.

### 4.2 Inference Characteristics

- Binary classification (biased / unbiased)
- Outputs both probability and confidence
- Inference latency ≈ 12 ms (CPU)

Importantly, predictions are framed as **signals, not decisions**, preserving human authority.

---

## 5. Explainability Engine as a Cognitive Layer

### 5.1 Separation of Concerns

Explainability is architected as a **first-class subsystem**, not a post-hoc visualization.

The explainability engine operates after prediction but is tightly coupled to the detection model through shared representations.

### 5.2 Dual-Channel Design

The engine integrates two complementary explanation pathways:

- **Attribution Channel (SHAP)**  
  → Supports causal attribution reasoning

- **Counterfactual Channel**  
  → Supports mental simulation and mitigation reasoning

This duality mirrors how humans reason under uncertainty: understanding why something happened and how it could be different.

### 5.3 Architectural Safeguards

To prevent explanation misuse:

- Explanations are local, not global
- Confidence is explicitly surfaced
- Explanations are optional, not forced

This avoids both opacity and overconfidence.

---

## 6. Interaction Layer (Human Interface)

### 6.1 Role of the Interface

The interface is not a presentation layer; it is a **reasoning workspace**.

Architecturally, it mediates:

- User input
- Model output
- Explanation depth
- Counterfactual exploration

### 6.2 Progressive Disclosure

The system implements progressive disclosure as an architectural pattern:

1. High-level bias signal
2. Localized token attribution
3. Counterfactual alternatives
4. Optional deep inspection

This prevents cognitive overload and discourages automation bias.

### 6.3 Accessibility and Responsiveness

- Keyboard navigation supported
- WCAG-compliant color contrast
- Sub-second model response (excluding explainability computation)

These constraints influenced upstream architectural choices, including model size and attribution method.

---

## 7. Evaluation and Feedback Loop

### 7.1 Multi-Axis Evaluation

The architecture explicitly supports evaluation across four axes:

1. Predictive performance
2. Fairness metrics
3. Explanation stability
4. Human interpretability

Evaluation components are modular, allowing metrics to evolve independently of the core model.

### 7.2 Feedback as Design Input

User feedback is treated as:

- An architectural signal
- Not merely usability noise

Insights from pilot users informed:

- Visualization choices
- Confidence calibration
- Explanation granularity

---

## 8. Modularity and Extensibility

BiasGuard Pro is designed to be extensible without architectural refactoring.

### Supported Extensions

- New bias dimensions (e.g., age, race)
- Multilingual models
- Alternative explanation methods
- API-based deployment

### Explicit Non-Goals

- Fully automated fairness enforcement
- Universal bias detection
- Black-box deployment

These boundaries are intentional and documented.

---

## 9. Failure Modes and Tradeoffs

### 9.1 Domain Specialization

The system is optimized for career and professional discourse.

Performance degradation outside this domain is an accepted tradeoff, not a flaw.

### 9.2 Binary Bias Framing

Binary classification simplifies interaction and explanation but cannot capture full social nuance.

This limitation is surfaced to users and mitigated through human review.

---

## 10. Architectural Summary

BiasGuard Pro's architecture operationalizes the principle that:

> Fairness is not a property of models alone, but of systems that support human reasoning.

By combining:

- Lightweight yet expressive modeling
- First-class explainability
- Interaction-aware design
- Explicit human oversight

the system moves fairness auditing from post-hoc analysis to **interactive, participatory practice**.
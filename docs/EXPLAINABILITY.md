# Explainability Design: BiasGuard Pro

## 1. Purpose of Explainability in BiasGuard Pro

In BiasGuard Pro, explainability is not treated as a transparency add-on, but as a **cognitive support mechanism** designed to help humans reason about algorithmic bias under uncertainty.

The system explicitly rejects the assumption that:

> "If a model is interpretable, users will automatically understand or trust it."

Instead, explainability is designed to:

- Support causal attribution
- Enable counterfactual reasoning
- Prevent automation bias
- Facilitate reflective human oversight

This design philosophy aligns with research in Human-Centered AI (HCAI), Explainable AI (XAI), and fairness-aware human–AI interaction.

---

## 2. Design Principles

BiasGuard Pro's explainability layer is guided by four principles:

### 2.1 Explanation ≠ Understanding

- Raw explanations (e.g., feature weights) do not guarantee human understanding
- Explanations must be interpretable, contextualized, and actionable

### 2.2 Support Human Reasoning, Not Model Introspection

The goal is not to expose internal model mechanics, but to help users answer:

- Why was this flagged?
- What linguistic elements matter?
- How could this be changed?

### 2.3 Progressive Disclosure Over Full Transparency

- Presenting all explanations at once increases cognitive load and over-trust
- BiasGuard Pro reveals explanations incrementally, based on user intent

### 2.4 Human Judgment Remains Central

- Explanations are advisory, not normative
- Final judgments about bias are explicitly left to the human user

---

## 3. Multi-Modal Explanation Architecture

BiasGuard Pro integrates two complementary explanation modalities, each aligned with a distinct cognitive process:

| Explanation Type | Cognitive Role | Question Answered |
|-----------------|----------------|-------------------|
| SHAP Attributions | Attributional reasoning | "Why was this flagged?" |
| Counterfactuals | Mental simulation | "What would change this?" |

These modalities are intentionally combined rather than offered in isolation.

---

## 4. SHAP-Based Token Attribution

### 4.1 Rationale

SHAP (SHapley Additive exPlanations) is used to provide localized, token-level attribution for bias predictions.

This supports attribution under ambiguity, a key cognitive challenge in bias auditing where:

- Bias may be implicit
- Linguistic cues are subtle
- Intent is unclear

### 4.2 What SHAP Explains (and What It Does Not)

**SHAP explanations in BiasGuard Pro indicate:**

- Which tokens contributed most to the model's bias prediction
- The direction and relative magnitude of contribution

**They do not claim:**

- Causality in a philosophical sense
- Ground truth about social harm
- Author intent

### 4.3 Presentation Strategy

- Token-level heatmaps overlaid directly on text
- Soft color gradients instead of discrete labels
- Highlighting limited to high-impact tokens by default

This design avoids visual overload while preserving interpretive value.

---

## 5. Counterfactual Explanations

### 5.1 Rationale

Understanding bias often requires mental simulation:

> "What if this were phrased differently?"

Counterfactual explanations support this by showing minimal, semantically coherent changes that alter the model's judgment.

### 5.2 Generation Process

1. Identify high-impact tokens via SHAP
2. Apply controlled substitutions using:
   - Gender-neutral terms
   - Profession-neutral phrasing
3. Re-evaluate bias probability
4. Refine fluency using a lightweight language model

Each counterfactual is validated to ensure:

- Reduced bias score
- Preserved semantic intent
- Linguistic coherence

### 5.3 Role in Bias Mitigation

Counterfactuals are **suggestive, not prescriptive**.

They demonstrate how language choices influence model behavior, enabling users to:

- Learn bias-sensitive phrasing
- Debug recommendation prompts
- Explore alternative formulations

---

## 6. Progressive Disclosure and Interaction Design

### 6.1 Explanation Layers

BiasGuard Pro follows a layered explanation model:

1. Bias score + confidence
2. Token-level highlights
3. Counterfactual alternatives
4. (Optional) deeper inspection for developers

Users explicitly choose how far to explore.

### 6.2 Trust Calibration

This interaction model is designed to:

- Reduce blind trust in automated outputs
- Avoid defensive rejection of flagged cases
- Encourage reflective engagement

Explanations are framed as **evidence, not verdicts**.

---

## 7. Explanation Stability and Reliability

To ensure explanations support consistent reasoning:

- SHAP attribution stability is evaluated across perturbations
- Mean cosine similarity ≈ 0.91
- Inconsistent explanations are treated as a failure mode

Stable explanations are essential for:

- Forming mental models
- Repeated auditing
- Human learning over time

---

## 8. Intended and Non-Intended Use

### Intended Use

- Bias auditing in career recommendation systems
- Educational exploration of gendered language
- Human-in-the-loop fairness workflows

### Non-Intended Use

- Automated content moderation
- Enforcement or punitive decision-making
- Replacement of human ethical judgment
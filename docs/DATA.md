# Dataset Documentation: BiasGuard Pro

## 1. Overview

BiasGuard Pro is trained and evaluated on a hybrid dataset that combines large-scale real-world professional text with carefully curated synthetic data. This design reflects a deliberate tradeoff between ecological validity, controlled bias isolation, and ethical data stewardship.

The dataset is constructed to support gendered stereotype detection in career recommendation and professional discourse, rather than general-purpose bias detection.

## 2. Data Sources

The full dataset consists of three components, each serving a distinct methodological role:

| Source | Role | Used for Training | Used for Evaluation |
|--------|------|-------------------|---------------------|
| Synthetic Career Text (GPT-4) | Controlled stereotype isolation | ✅ Yes | ✅ Yes |
| BiasBios | Real-world professional language | ✅ Yes | ✅ Yes |
| StereoSet (Gender subset) | Out-of-domain generalization | ❌ No | ✅ Yes |

## 3. Synthetic Career Recommendation Dataset

### 3.1 Motivation

Real-world datasets often conflate bias signals with unrelated linguistic variation, making it difficult to isolate why a model flags content as biased. To address this, BiasGuard Pro includes a synthetic, paired dataset designed to:

- Explicitly encode common gendered stereotypes
- Control for semantic content
- Enable attribution-level interpretability analysis

### 3.2 Generation Process

- **Model used:** GPT-4
- **Prompting strategy:** Structured prompts generating paired sentences with identical semantics but differing only in gendered framing
- **Domains covered:**
  - Nurturing / caregiving roles
  - Leadership and decisiveness
  - Technical and analytical aptitude

**Example Pair:**

- **Biased:** "Women with your compassion should consider nursing."
- **Neutral:** "Your compassion makes you a strong fit for healthcare roles."

This pairing strategy minimizes confounds while isolating gendered framing as the primary variable.

### 3.3 Dataset Composition

| Attribute | Value |
|-----------|-------|
| Total samples | 640 |
| Biased samples | 320 |
| Neutral samples | 320 |
| Gender references | Balanced |
| Stereotype categories | Balanced |

All synthetic samples were manually reviewed to ensure:

- Linguistic plausibility
- Absence of prompt artifacts
- Alignment with real professional discourse

## 4. BiasBios Dataset (Real-World Data)

### 4.1 Description

BiasBios is a large-scale corpus of professional biographies annotated for gender-profession associations. It captures naturally occurring bias patterns in occupational language at scale.

- **Approximate size:** 396,000 biographies
- **Domain:** Professional and career-related text
- **Gender inference:** Derived from names and pronouns (as provided in the original dataset)

### 4.2 Label Construction

BiasBios provides continuous bias association scores rather than binary labels. To align with BiasGuard Pro's operational setting, labels were derived using a threshold-based binarization:

- Top 20% of gender-association scores → Biased
- Bottom 20% → Unbiased
- Middle 60% → Treated as neutral for binarized classification, focusing the learning objective on clearer bias signals.

This choice focuses training on clear bias signals, reducing noise from ambiguous stylistic variation.

### 4.3 Rationale

This approach prioritizes:

- High-confidence bias examples
- Reduced label ambiguity
- Clear interpretability during explanation analysis

## 5. Dataset Integration and Splitting

### 5.1 Combined Corpus

The synthetic dataset was concatenated with BiasBios to form a unified training corpus.

| Component | Approx. Samples |
|-----------|----------------|
| BiasBios | ~396,000 |
| Synthetic | 640 |
| Total | ~396,640 |

Synthetic data represents a small but strategically important fraction of the corpus, acting as a bias-control anchor rather than a distributional driver.

### 5.2 Train / Validation / Test Split

A stratified split was applied to preserve class balance, gender indicators, and profession diversity:

- **Training:** 72%
- **Validation:** 18%
- **Test:** 10%

The held-out test set contains approximately 39,600 samples, including both real and synthetic data.

## 6. External Evaluation Dataset: StereoSet

### 6.1 Purpose

The gender subset of StereoSet is used exclusively for evaluation, not training.

Its role is to measure:

- Cross-domain generalization
- Sensitivity to general-language stereotypes outside professional contexts

### 6.2 Characteristics

| Attribute | Value |
|-----------|-------|
| Language | English |
| Domain | General language |
| Size | 765 sentences |
| Bias type | Stereotypical associations |

### 6.3 Interpretation Caveat

Performance degradation on StereoSet is expected and intentional. BiasGuard Pro is domain-specialized for career and professional discourse, not general stereotype detection.

## 7. Preprocessing

- **Tokenization:** DistilBERT tokenizer
- **Maximum sequence length:** 512 tokens
- **No demographic inference** beyond dataset-provided indicators
- **No personally identifiable information (PII)** introduced or inferred

## 8. Ethical Considerations

### 8.1 Privacy and Consent

- No new personal data was collected
- Synthetic data was explicitly used to reduce reliance on sensitive real-world examples
- BiasBios and StereoSet are used under their original research licenses

### 8.2 Scope Limitations

The dataset:

- Focuses on binary gender
- Operates in English
- Reflects primarily Western professional norms

These limitations are documented and intentional, not implicit assumptions.

### 8.3 Intended Use

This dataset is designed for:

- Bias auditing
- Fairness research
- Explainability studies

It is **not** intended for:

- Automated decision enforcement
- Ranking or penalization of individuals
- Standalone deployment without human oversight
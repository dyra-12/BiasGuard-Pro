"""SHAP-based explanation utilities for BiasGuard Pro.

This module exposes a thin wrapper around SHAP's text explainer that
integrates with the project's `BiasDetector` model. It provides a
convenience API to obtain top contributing tokens for the model's
bias prediction, and a small fallback analyzer when SHAP is unavailable.

The implementation includes helpers to merge tokenizer subword scores
back into readable words.
"""

import re
from typing import List, Tuple

import numpy as np
import shap
import torch

from .bias_detector import BiasDetector


class SHAPExplainer:
    """Wrapper around SHAP explainer for BiasDetector.

    Args:
        model_path: Path where the BiasDetector model/tokenizer are located
            (or a HuggingFace model id). The underlying `BiasDetector` will be
            initialized from this path.

    Attributes:
        detector: Instance of :class:`BiasDetector` used to get model preds.
        explainer: SHAP Explainer instance or None if initialization failed.
    """

    def __init__(self, model_path: str = "."):
        print("Loading SHAP explainer...")
        self.detector = BiasDetector(model_path)

        # SHAP expects a prediction function that accepts a list of texts and
        # returns a 2D array of shape (n_samples, n_outputs). The BiasDetector
        # returns richer dicts, so we wrap its API to expose only the
        # scalar bias probability expected by SHAP.
        def model_predict(texts):
            """Prediction wrapper used by SHAP.

            Accepts a single string or a list of strings and returns a NumPy
            array shaped (n_samples, 1) containing the bias probability.
            """

            if isinstance(texts, str):
                texts = [texts]
            texts = [str(t) for t in texts]

            # Use the bias detector's probability output
            results = [self.detector.predict_bias(text) for text in texts]
            # Return shape (n_samples, 1) for compatibility with SHAP
            # SHAP's Explainer expects a 2D array (n_samples, n_outputs). Our
            # model produces a single scalar per example (bias probability),
            # so we expose it as a column vector. Keeping the 2D shape avoids
            # version-dependent SHAP behavior where SHAP sometimes assumes
            # multi-output models.
            return np.array([[result["bias_probability"]] for result in results])

        self.model_predict = model_predict

        try:
            # Create a text masker that knows about tokenization. Note: the
            # exact masker API can vary by SHAP version; here we pass the
            # tokenizer so SHAP can convert masked token positions back to
            # text inputs when computing attributions.
            masker = shap.maskers.Text(tokenizer=self.detector.tokenizer)
            # Construct the explainer using our prediction wrapper. The
            # resulting object produces per-token attribution values that
            # must be post-processed below.
            self.explainer = shap.Explainer(self.model_predict, masker=masker)
            print("✅ SHAP Explainer initialized successfully!")
        except Exception as e:
            # If SHAP or the underlying tokenizer is incompatible in the
            # current environment, fall back gracefully.
            print(f"❌ Error initializing SHAP: {e}")
            self.explainer = None

    def get_shap_values(
        self, text: str, max_evals: int = 500
    ) -> List[Tuple[str, float]]:
        """Return a ranked list of (token, score) tuples explaining `text`.

        The function returns the top contributing tokens (by absolute SHAP
        value) to the model's bias prediction. If SHAP is unavailable or
        explainer evaluation fails, a lightweight keyword-based fallback is
        used instead.

        Args:
            text: Input string to explain.
            max_evals: Maximum number of SHAP evaluations (passed to SHAP).

        Returns:
            A list of (word, score) tuples sorted by absolute contribution
            (descending). Scores are floats (positive or negative).
        """

        if self.explainer is None:
            return self._fallback_analysis(text)

        try:
            # SHAP can return different shaped arrays depending on the
            # model/predictor shape; evaluate once and normalize to a 1D
            # sequence of per-token contributions for the single output.
            shap_values = self.explainer([text], max_evals=max_evals)

            # SHAP sometimes returns shape (1, n_tokens, 1) for a single
            # scalar output, or (1, n_tokens) in other versions. Handle both.
            # The explainer's `.values` holds attribution scores per token.
            # Normalizing to a 1D array of length n_tokens makes downstream
            # merging logic independent of SHAP version quirks.
            if len(shap_values.values.shape) == 3:
                biased_values = shap_values.values[0, :, 0]
            else:
                biased_values = shap_values.values[0, :]

            tokens = shap_values.data[0]

            # Merge subword tokens (e.g., '##ing') back to whole words
            combined_scores = self._combine_subword_scores(tokens, biased_values)

            # Filter out special tokens and very short tokens for readability
            filtered_scores = [
                (w, float(s))
                for w, s in combined_scores
                if w.strip() and w not in ["[cls]", "[sep]", "[pad]"] and len(w) > 1
            ]
            filtered_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            return filtered_scores[:10]

        except Exception as e:
            print(f"❌ SHAP calculation failed: {e}")
            return self._fallback_analysis(text)

    def _combine_subword_scores(
        self, tokens: List[str], scores: np.ndarray
    ) -> List[Tuple[str, float]]:
        """Combine WordPiece/BPE subword token scores into whole-word scores.

        Many tokenizers split words into subwords prefixed with '##' (or
        similar markers). This helper merges consecutive subword tokens
        belonging to the same original word and averages their SHAP scores.

        Args:
            tokens: Sequence of token strings produced by the tokenizer.
            scores: 1D NumPy array of per-token SHAP scores.

        Returns:
            List of (word, averaged_score) tuples in token order.
        """

        combined = []
        current_word, current_score, token_count = "", 0.0, 0

        for token, score in zip(tokens, scores):
            # Many tokenizers (WordPiece) mark subword continuations with
            # a '##' prefix. Some BPE/tokenizers use other markers (e.g.
            # leading '▁' or none). Here we only handle the common '##'
            # pattern used by HuggingFace's WordPiece-based models. The goal
            # is to reassemble the original human-readable word and average
            # the SHAP scores across its subpieces.
            if token.startswith("##"):
                # Continuation: append the piece (without the prefix) and
                # accumulate the score. We count pieces to compute an
                # average attribution for the whole word.
                current_word += token[2:]
                current_score += score
                token_count += 1
            else:
                # New token begins. If we were accumulating subwords,
                # flush the assembled word first.
                if current_word:
                    combined.append((current_word, current_score / token_count))
                # Start accumulation for the new token. Note: we treat the
                # non-subword token as the first piece of the new word.
                current_word, current_score, token_count = token, score, 1

        # Flush any trailing word
        if current_word:
            combined.append((current_word, current_score / token_count))

        # Clean up artifacts like '##' and whitespace. This also makes the
        # output more consistent across tokenizer types.
        return [(word.replace("##", "").strip(), score) for word, score in combined]

    def _fallback_analysis(self, text: str) -> List[Tuple[str, float]]:
        """Lightweight keyword-based analyzer used when SHAP is unavailable.

        This function scans the text for a small curated set of bias-related
        tokens and assigns heuristic scores. It exists to provide a best-effort
        explanation in constrained environments where SHAP cannot run.
        """

        # Simple fallback based on known bias keywords
        bias_keywords = {
            "women",
            "men",
            "female",
            "male",
            "she",
            "he",
            "her",
            "his",
            "naturally",
            "should",
            "emotional",
            "aggressive",
            "compassionate",
            "nurse",
            "engineer",
            "secretary",
            "teacher",
            "better",
            "too",
        }

        words = re.findall(r"\b\w+\b", text.lower())
        scores = []

        for word in words:
            if word in bias_keywords:
                # Heuristic scoring: high weight for overt demographic terms
                # and prescriptive language, medium for personality labels,
                # and lower weight for less-specific vocabulary. These are
                # intentionally simple; they exist only to provide a readable
                # fallback when SHAP is not available.
                if word in ["women", "men", "naturally", "should"]:
                    scores.append((word, 0.8))
                elif word in ["emotional", "aggressive", "compassionate"]:
                    scores.append((word, 0.6))
                else:
                    scores.append((word, 0.4))

        return sorted(scores, key=lambda x: x[1], reverse=True)[:8]

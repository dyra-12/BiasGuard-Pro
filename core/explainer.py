import shap
import torch
import numpy as np
import re
from typing import List, Tuple
from .bias_detector import BiasDetector

class SHAPExplainer:
    def __init__(self, model_path: str = '.'):
        print("Loading SHAP explainer...")
        self.detector = BiasDetector(model_path)
        
        def model_predict(texts):
            if isinstance(texts, str):
                texts = [texts]
            texts = [str(t) for t in texts]
            
            # Use the bias detector's probability
            results = [self.detector.predict_bias(text) for text in texts]
            return np.array([[result['bias_probability']] for result in results])

        self.model_predict = model_predict

        try:
            masker = shap.maskers.Text(tokenizer=self.detector.tokenizer)
            self.explainer = shap.Explainer(self.model_predict, masker=masker)
            print("✅ SHAP Explainer initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing SHAP: {e}")
            self.explainer = None

    def get_shap_values(self, text: str, max_evals: int = 500) -> List[Tuple[str, float]]:
        if self.explainer is None:
            return self._fallback_analysis(text)
        
        try:
            shap_values = self.explainer([text], max_evals=max_evals)
            
            if len(shap_values.values.shape) == 3:
                biased_values = shap_values.values[0, :, 0]
            else:
                biased_values = shap_values.values[0, :]
                
            tokens = shap_values.data[0]
            
            combined_scores = self._combine_subword_scores(tokens, biased_values)
            filtered_scores = [
                (w, float(s)) for w, s in combined_scores
                if w.strip() and w not in ['[cls]', '[sep]', '[pad]'] and len(w) > 1
            ]
            filtered_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            return filtered_scores[:10]
            
        except Exception as e:
            print(f"❌ SHAP calculation failed: {e}")
            return self._fallback_analysis(text)

    def _combine_subword_scores(self, tokens: List[str], scores: np.ndarray) -> List[Tuple[str, float]]:
        combined = []
        current_word, current_score, token_count = "", 0.0, 0
        
        for token, score in zip(tokens, scores):
            if token.startswith("##"):
                current_word += token[2:]
                current_score += score
                token_count += 1
            else:
                if current_word:
                    combined.append((current_word, current_score / token_count))
                current_word, current_score, token_count = token, score, 1
        
        if current_word:
            combined.append((current_word, current_score / token_count))
            
        return [(word.replace('##', '').strip(), score) for word, score in combined]

    def _fallback_analysis(self, text: str) -> List[Tuple[str, float]]:
        # Simple fallback based on known bias keywords
        bias_keywords = {
            'women', 'men', 'female', 'male', 'she', 'he', 'her', 'his',
            'naturally', 'should', 'emotional', 'aggressive', 'compassionate',
            'nurse', 'engineer', 'secretary', 'teacher', 'better', 'too'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        scores = []
        
        for word in words:
            if word in bias_keywords:
                if word in ['women', 'men', 'naturally', 'should']:
                    scores.append((word, 0.8))
                elif word in ['emotional', 'aggressive', 'compassionate']:
                    scores.append((word, 0.6))
                else:
                    scores.append((word, 0.4))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:8]
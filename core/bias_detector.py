import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import Dict, List
import os

class BiasDetector:
    def __init__(self, model_path: str = '.'):
        print("Loading bias detection model...")
        
        try:
            # Try to load from specified path first
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            print(f"✅ Loaded model from: {model_path}")
        except Exception as e:
            print(f"❌ Could not load model from {model_path}: {e}")
            print("🔄 Falling back to default model...")
            # Fallback to a base model
            self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            print("✅ Loaded default model")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()
        print(f"Bias detector loaded on: {self.device}")
    
    def predict_bias(self, text: str) -> Dict:
        """Predict bias probability for a single text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            bias_prob = probs[0, 1].item()
        
        return {
            'bias_probability': bias_prob,
            'classification': 'BIASED' if bias_prob > 0.5 else 'NEUTRAL',
            'confidence': max(bias_prob, 1 - bias_prob)
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict bias for multiple texts"""
        return [self.predict_bias(text) for text in texts]

    def predict_batch_batched(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """Predict bias for multiple texts using batched model inference for speed.

        Returns the same dict per item as `predict_bias`.
        """
        results: List[Dict] = []
        if not texts:
            return results

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            for j in range(len(batch_texts)):
                bias_prob = probs[j, 1].item()
                results.append({
                    'bias_probability': bias_prob,
                    'classification': 'BIASED' if bias_prob > 0.5 else 'NEUTRAL',
                    'confidence': max(bias_prob, 1 - bias_prob)
                })

        return results
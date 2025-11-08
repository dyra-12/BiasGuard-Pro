import os
import sys
from pathlib import Path

from bias_detector import BiasDetector
from explainer import SHAPExplainer
from counterfactuals import CounterfactualGenerator
from typing import Dict, List
import json

class BiasGuardPro:
    def __init__(self, model_path: str = None):
        print("🚀 Initializing BiasGuard Pro...")
        
        # Auto-detect model path
        if model_path is None:
            model_path = self._auto_detect_model_path()
        
        print(f"📁 Using model path: {model_path}")
        self.detector = BiasDetector(model_path)
        self.explainer = SHAPExplainer(model_path)
        self.counterfactual_generator = CounterfactualGenerator()
        print("✅ BiasGuard Pro initialized successfully!\n")
    
    def _auto_detect_model_path(self) -> str:
        """Auto-detect where model files are located"""
        possible_paths = [
            '.',  # Current directory
            './models',
            './model',
            '../models',
            '../model'
        ]
        
        # Check for model files in each path
        model_extensions = ('.safetensors', '.bin', '.json')
        for path in possible_paths:
            if os.path.exists(path):
                files = os.listdir(path)
                if any(f.endswith(model_extensions) for f in files):
                    print(f"✅ Found model files in: {path}")
                    return path
        
        # If no model files found, use default Hugging Face model as fallback
        print("⚠️  No local model files found. Using default model...")
        return "distilbert-base-uncased"
    
    def analyze_text(self, text: str) -> Dict:
        """Complete analysis of text for bias"""
        print(f"🔍 Analyzing: '{text}'")
        
        # Get bias prediction
        bias_result = self.detector.predict_bias(text)
        
        # Get SHAP explanations
        shap_results = self.explainer.get_shap_values(text)
        print(f"   Top biased words: {[w for w, s in shap_results[:3]]}")
        
        # Generate counterfactuals
        counterfactuals = self.counterfactual_generator.generate_counterfactuals(text, shap_results)
        
        return {
            'text': text,
            'bias_probability': bias_result['bias_probability'],
            'bias_class': bias_result['classification'],
            'confidence': bias_result['confidence'],
            'top_biased_words': [w for w, s in shap_results[:3]],
            'shap_scores': shap_results[:10],
            'counterfactuals': counterfactuals
        }
    
    def load_examples(self) -> Dict:
        """Load example texts from assets"""
        examples_path = os.path.join('assets', 'examples.json')
        if os.path.exists(examples_path):
            with open(examples_path, 'r') as f:
                return json.load(f)
        return {"biased_examples": [], "neutral_examples": []}

def main():
    # Test the complete system with auto-detection
    bias_guard = BiasGuardPro()  # No path specified - auto-detect
    
    # Load examples
    examples = bias_guard.load_examples()
    
    # Combine biased and neutral examples for testing
    test_cases = examples.get('biased_examples', []) + examples.get('neutral_examples', [])
    
    # If no examples found, use default test cases
    if not test_cases:
        test_cases = [
            "Women should be nurses because they are compassionate.",
            "Men are naturally better at engineering roles.",
            "The female secretary was very emotional today.",
            "He needs to be more aggressive to succeed in business.",
            "Women are too emotional for leadership positions.",
            "People with technical skills excel in engineering roles.",
            "The candidate demonstrated strong problem-solving abilities.",
            "Effective leadership requires emotional intelligence and strategic thinking."
        ]
    
    print("🧪 Testing BiasGuard Pro System")
    print("=" * 60)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n📋 Example {i}/{len(test_cases)}")
        result = bias_guard.analyze_text(text)
        
        print(f"📝 Original: {result['text']}")
        print(f"🎯 Bias Probability: {result['bias_probability']:.3f} ({result['bias_class']})")
        print(f"🔍 Top Biased Words: {result['top_biased_words']}")
        print("🔄 Counterfactuals:")
        for j, cf in enumerate(result['counterfactuals'], 1):
            print(f"   {j}. {cf}")
        
        if i < len(test_cases):
            print("-" * 60)

if __name__ == "__main__":
    main()
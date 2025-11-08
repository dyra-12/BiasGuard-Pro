"""
Cross-Dataset Evaluation Script
Evaluates models on BiasBios and StereoSet datasets for binary classification
"""

import torch
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import argparse


class TextClassificationDataset(Dataset):
    """Custom Dataset for text classification tasks"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class CrossDatasetEvaluator:
    """Main class for cross-dataset evaluation"""
    
    def __init__(self, model_path: str, batch_size: int = 16, max_length: int = 128):
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
        # Create results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def load_model(self) -> bool:
        """Load model and tokenizer from local path"""
        try:
            print(f"Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model = self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_biasbios_data(self, file_path: str) -> Tuple[List[str], List[int]]:
        """Load BiasBios dataset"""
        print(f"Loading BiasBios data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Adjust column names based on your dataset
        text_col = 'text'
        label_col = 'label_binary'
        
        if text_col not in df.columns:
            # Try to find alternative column names
            possible_text_cols = ['text', 'hard_text', 'sentence']
            text_col = next((col for col in possible_text_cols if col in df.columns), None)
        
        if label_col not in df.columns:
            possible_label_cols = ['label_binary', 'bias_score', 'label']
            label_col = next((col for col in possible_label_cols if col in df.columns), None)
        
        if text_col is None or label_col is None:
            raise ValueError(f"Could not find appropriate columns. Available columns: {df.columns.tolist()}")
        
        texts = df[text_col].fillna('').astype(str).tolist()
        labels = df[label_col].astype(int).tolist()
        
        print(f"Loaded {len(texts)} samples from BiasBios")
        return texts, labels
    
    def load_stereoset_data(self, file_path: str) -> Tuple[List[str], List[int]]:
        """Load StereoSet dataset"""
        print(f"Loading StereoSet data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Adjust column names based on your dataset
        text_col = 'text'
        label_col = 'binary_label'
        
        if text_col not in df.columns:
            possible_text_cols = ['text', 'sentence', 'context']
            text_col = next((col for col in possible_text_cols if col in df.columns), None)
        
        if label_col not in df.columns:
            possible_label_cols = ['binary_label', 'gold_label', 'label']
            label_col = next((col for col in possible_label_cols if col in df.columns), None)
        
        if text_col is None or label_col is None:
            raise ValueError(f"Could not find appropriate columns. Available columns: {df.columns.tolist()}")
        
        texts = df[text_col].fillna('').astype(str).tolist()
        labels = df[label_col].astype(int).tolist()
        
        print(f"Loaded {len(texts)} samples from StereoSet")
        return texts, labels
    
    def evaluate_dataset(self, texts: List[str], labels: List[int], dataset_name: str) -> Dict[str, Any]:
        """Evaluate model on a single dataset"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded. Call load_model() first.")
        
        print(f"Evaluating on {dataset_name}...")
        
        # Create dataset and dataloader
        dataset = TextClassificationDataset(texts, labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels_batch.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='binary')
        report = classification_report(true_labels, predictions, output_dict=False)
        report_dict = classification_report(true_labels, predictions, output_dict=True)
        
        print(f"\n{'='*50}")
        print(f"Results for {dataset_name}")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"\nClassification Report:")
        print(report)
        
        return {
            'predictions': predictions,
            'true_labels': true_labels,
            'texts': texts,
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'classification_report_dict': report_dict
        }
    
    def save_results(self, results: Dict[str, Any], dataset_name: str):
        """Save predictions and metrics to files"""
        if results is None:
            return
        
        # Sanitize dataset name for filename
        safe_name = dataset_name.replace(" ", "_").lower()
        
        # Save predictions
        results_df = pd.DataFrame({
            'text': results['texts'],
            'true_label': results['true_labels'],
            'predicted_label': results['predictions']
        })
        
        predictions_filename = self.results_dir / f"{safe_name}_predictions.csv"
        results_df.to_csv(predictions_filename, index=False)
        
        # Save metrics
        metrics = {
            'dataset': dataset_name,
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score'],
            'classification_report': results['classification_report_dict']
        }
        
        metrics_filename = self.results_dir / f"{safe_name}_metrics.json"
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Saved predictions to: {predictions_filename}")
        print(f"Saved metrics to: {metrics_filename}")
    
    def run_evaluation(self, biasbios_path: str, stereoset_path: str):
        """Run full cross-dataset evaluation"""
        print("Starting Cross-Dataset Evaluation...")
        print(f"Using device: {self.device}")
        
        # Load model
        if not self.load_model():
            return
        
        # Load datasets
        try:
            biasbios_texts, biasbios_labels = self.load_biasbios_data(biasbios_path)
            stereoset_texts, stereoset_labels = self.load_stereoset_data(stereoset_path)
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return
        
        # Evaluate on both datasets
        biasbios_results = self.evaluate_dataset(biasbios_texts, biasbios_labels, "BiasBios Test")
        stereoset_results = self.evaluate_dataset(stereoset_texts, stereoset_labels, "StereoSet Gender")

        # ensure dataset name is recorded inside results for comparison table
        biasbios_results['dataset'] = "BiasBios Test"
        stereoset_results['dataset'] = "StereoSet Gender"

        # Save results
        self.save_results(biasbios_results, "biasbios_test")
        self.save_results(stereoset_results, "stereoset_gender")

        # Save combined comparison table
        try:
            self.save_comparison_table([biasbios_results, stereoset_results])
            print(f"Saved cross-dataset comparison to: {self.results_dir / 'crossdataset_results.csv'}")
        except Exception as e:
            print(f"Failed to save cross-dataset comparison table: {e}")

        # Print summary
        self.print_summary(biasbios_results, stereoset_results)

    def _row_from_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single evaluation's results into a flat row for the comparison table."""
        report_dict = results.get('classification_report_dict', {})
        # Class labels might be strings like '0' and '1'
        cls0 = report_dict.get('0', {})
        cls1 = report_dict.get('1', {})
        macro = report_dict.get('macro avg', {}) or report_dict.get('macro_avg', {})
        weighted = report_dict.get('weighted avg', {}) or report_dict.get('weighted_avg', {})

        support_0 = int(cls0.get('support', 0)) if cls0 else 0
        support_1 = int(cls1.get('support', 0)) if cls1 else 0

        row = {
            'dataset': results.get('dataset', ''),
            'accuracy': float(results.get('accuracy', np.nan)),
            'f1_score': float(results.get('f1_score', np.nan)),
            'precision_0': float(cls0.get('precision', np.nan)) if cls0 else np.nan,
            'recall_0': float(cls0.get('recall', np.nan)) if cls0 else np.nan,
            'f1_0': float(cls0.get('f1-score', np.nan)) if cls0 else np.nan,
            'support_0': support_0,
            'precision_1': float(cls1.get('precision', np.nan)) if cls1 else np.nan,
            'recall_1': float(cls1.get('recall', np.nan)) if cls1 else np.nan,
            'f1_1': float(cls1.get('f1-score', np.nan)) if cls1 else np.nan,
            'support_1': support_1,
            'macro_precision': float(macro.get('precision', np.nan)) if macro else np.nan,
            'macro_recall': float(macro.get('recall', np.nan)) if macro else np.nan,
            'macro_f1': float(macro.get('f1-score', np.nan)) if macro else np.nan,
            'weighted_precision': float(weighted.get('precision', np.nan)) if weighted else np.nan,
            'weighted_recall': float(weighted.get('recall', np.nan)) if weighted else np.nan,
            'weighted_f1': float(weighted.get('f1-score', np.nan)) if weighted else np.nan,
            'total_support': support_0 + support_1
        }

        return row

    def save_comparison_table(self, results_list: List[Dict[str, Any]], filename: str = 'crossdataset_results.csv'):
        """Build a comparison table from multiple evaluation results and save as CSV.

        The function expects each results dict to contain 'classification_report_dict',
        'accuracy', 'f1_score' and optionally 'dataset'.
        """
        rows = []
        for r in results_list:
            # Ensure we have a dataset label for clarity
            if 'dataset' not in r or not r.get('dataset'):
                # Attempt to infer from saved metrics JSON if present
                r['dataset'] = r.get('classification_report_dict', {}).get('dataset', '')
            rows.append(self._row_from_report(r))

        df = pd.DataFrame(rows)
        out_path = self.results_dir / filename
        df.to_csv(out_path, index=False)
    
    def print_summary(self, biasbios_results: Dict[str, Any], stereoset_results: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("CROSS-DATASET EVALUATION SUMMARY")
        print("="*60)
        
        if biasbios_results is not None:
            print(f"\nBiasBios Test Results:")
            print(f"  Accuracy: {biasbios_results['accuracy']:.4f}")
            print(f"  F1 Score: {biasbios_results['f1_score']:.4f}")
        
        if stereoset_results is not None:
            print(f"\nStereoSet Gender Results:")
            print(f"  Accuracy: {stereoset_results['accuracy']:.4f}")
            print(f"  F1 Score: {stereoset_results['f1_score']:.4f}")
        
        print("\n" + "="*60)


def main():
    """Main function to run evaluation from command line"""
    parser = argparse.ArgumentParser(description='Cross-Dataset Evaluation for Bias Detection')
    # sensible defaults for local repo layout; users can override as needed
    default_model = os.path.join('models')
    default_biasbios = os.path.join('data', 'processed', 'biasbios_test.csv')
    default_stereoset = os.path.join('data', 'raw', 'stereoset_gender.csv')

    parser.add_argument('--model_path', type=str, required=False, default=default_model,
                        help=f'Path to the model directory (default: {default_model})')
    parser.add_argument('--biasbios_path', type=str, required=False, default=default_biasbios,
                        help=f'Path to BiasBios test CSV (default: {default_biasbios})')
    parser.add_argument('--stereoset_path', type=str, required=False, default=default_stereoset,
                        help=f'Path to StereoSet CSV (default: {default_stereoset})')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = CrossDatasetEvaluator(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Run evaluation
    evaluator.run_evaluation(
        biasbios_path=args.biasbios_path,
        stereoset_path=args.stereoset_path
    )


if __name__ == "__main__":
    main()
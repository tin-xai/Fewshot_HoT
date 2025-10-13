#!/usr/bin/env python3
"""
Rule-Based Evaluation Script
============================

Extracts answers from curly brackets {} and compares to ground truth.
No LLM needed - just pattern matching and comparison.

Usage:
    python rule_based_eval.py --dataset date --model_output_file results/date/cot/fs_inst_gemini-2.0-flash-001_temp_10_full.csv
"""

import os
import sys
import re
import argparse
from typing import Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.load_dataset import DatasetLoader


class AnswerExtractor:
    """Extract answers from model outputs."""
    
    @staticmethod
    def extract_from_brackets(text: str) -> Optional[str]:
        """Extract answer from curly brackets {}."""
        if not text or not isinstance(text, str):
            return None
        
        # Find all matches of content in curly brackets
        matches = re.findall(r'\{([^}]+)\}', text)
        
        if matches:
            # Return the last match (usually the final answer)
            return matches[-1].strip()
        
        return None
    
    @staticmethod
    def normalize_answer(answer: Any) -> str:
        """Normalize answer for comparison."""
        if answer is None:
            return ""
        
        answer_str = str(answer).strip()
        
        # Remove common punctuation and extra whitespace
        answer_str = re.sub(r'\s+', ' ', answer_str)
        
        return answer_str.lower()


class AnswerComparator:
    """Compare extracted answers with ground truth."""
    
    @staticmethod
    def compare_dates(extracted: str, ground_truth: str) -> bool:
        """Compare date answers (handles different formats)."""
        # Normalize both dates
        ext_norm = re.sub(r'[^0-9/]', '', extracted)
        gt_norm = re.sub(r'[^0-9/]', '', str(ground_truth))
        
        if ext_norm == gt_norm:
            return True
        
        # Try to parse and compare date components
        date_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
        ]
        
        for pattern in date_patterns:
            ext_match = re.search(pattern, extracted)
            gt_match = re.search(pattern, str(ground_truth))
            
            if ext_match and gt_match:
                ext_parts = [int(x) for x in ext_match.groups()]
                gt_parts = [int(x) for x in gt_match.groups()]
                
                if ext_parts == gt_parts:
                    return True
        
        return False
    
    @staticmethod
    def compare_numbers(extracted: str, ground_truth: Any) -> bool:
        """Compare numeric answers."""
        try:
            # Remove commas and convert to float
            ext_num = float(re.sub(r'[,\s]', '', extracted))
            gt_num = float(re.sub(r'[,\s]', '', str(ground_truth)))
            
            # Use a small epsilon for floating point comparison
            return abs(ext_num - gt_num) < 1e-6
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def compare_exact(extracted: str, ground_truth: str) -> bool:
        """Exact string comparison (case-insensitive)."""
        return extracted.lower().strip() == str(ground_truth).lower().strip()
    
    @classmethod
    def compare(cls, extracted: Optional[str], ground_truth: Any, dataset: str = None) -> bool:
        """
        Compare extracted answer with ground truth.
        
        Args:
            extracted: Extracted answer from model output
            ground_truth: Ground truth answer
            dataset: Dataset name for dataset-specific comparison logic
            
        Returns:
            True if answers match, False otherwise
        """
        if extracted is None:
            return False
        
        # Try date comparison first if dataset is 'date'
        if dataset == 'date':
            if cls.compare_dates(extracted, ground_truth):
                return True
        
        # Try numeric comparison
        if cls.compare_numbers(extracted, ground_truth):
            return True
        
        # Try exact comparison
        if cls.compare_exact(extracted, ground_truth):
            return True
        
        # Check if ground truth is contained in extracted (for some datasets)
        if str(ground_truth).lower().strip() in extracted.lower():
            return True
        
        return False


class RuleBasedEvaluator:
    """Rule-based evaluator using pattern matching."""
    
    def __init__(self, dataset: str, base_data_path: str = 'data/data/'):
        self.dataset = dataset
        self.extractor = AnswerExtractor()
        self.comparator = AnswerComparator()
        
        # Get paths relative to this script's parent directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        config_path = os.path.join(parent_dir, 'configs', 'config.yaml')
        data_path = os.path.join(parent_dir, base_data_path) if not os.path.isabs(base_data_path) else base_data_path
        fewshot_path = os.path.join(parent_dir, 'fewshot_prompts')
        
        self.dataset_loader = DatasetLoader(
            config_path=config_path,
            base_data_path=data_path,
            base_few_shot_prompt_path=fewshot_path,
            dataset=dataset,
            data_mode='longest',
            num_samples=400
        )
    
    def evaluate_csv(self, csv_path: str) -> Tuple[pd.DataFrame, dict]:
        """
        Evaluate a CSV file with model outputs.
        
        Args:
            csv_path: Path to CSV file with columns: id, question, answer
            
        Returns:
            Tuple of (results_df, metrics_dict)
        """
        # Load model outputs
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} examples from {csv_path}")
        
        # Get ground truths
        ids = df['id'].tolist()
        ground_truths = self.dataset_loader.retrieve_gts(ids)
        print(f"Loaded {len(ground_truths)} ground truth answers")
        
        # Evaluate each example
        results = []
        correct_count = 0
        extracted_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            model_answer = row['answer']
            ground_truth = ground_truths[idx]
            
            # Extract answer from curly brackets
            extracted = self.extractor.extract_from_brackets(model_answer)
            
            if extracted:
                extracted_count += 1
            
            # Compare with ground truth
            is_correct = self.comparator.compare(extracted, ground_truth, self.dataset)
            
            if is_correct:
                correct_count += 1
            
            results.append({
                'id': row['id'],
                'question': row['question'],
                'model_answer': model_answer,
                'extracted_answer': extracted if extracted else "[NOT FOUND]",
                'ground_truth': ground_truth,
                'is_correct': is_correct
            })
        
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        total = len(results)
        accuracy = (correct_count / total * 100) if total > 0 else 0
        extraction_rate = (extracted_count / total * 100) if total > 0 else 0
        
        metrics = {
            'total': total,
            'correct': correct_count,
            'incorrect': total - correct_count,
            'extracted': extracted_count,
            'not_extracted': total - extracted_count,
            'accuracy': accuracy,
            'extraction_rate': extraction_rate
        }
        
        return results_df, metrics


def print_metrics(metrics: dict):
    """Print evaluation metrics."""
    print("\n" + "="*60)
    print("EVALUATION METRICS (RULE-BASED)")
    print("="*60)
    print(f"Total Examples:       {metrics['total']}")
    print(f"Answers Extracted:    {metrics['extracted']} ({metrics['extraction_rate']:.2f}%)")
    print(f"Answers Not Found:    {metrics['not_extracted']}")
    print(f"Correct:              {metrics['correct']}")
    print(f"Incorrect:            {metrics['incorrect']}")
    print(f"Accuracy:             {metrics['accuracy']:.2f}%")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Rule-Based Evaluation")
    parser.add_argument('--dataset', type=str, default='date',
                       help='Dataset name (e.g., date, GSM8K, etc.)')
    parser.add_argument('--model_output_file', type=str, required=True,
                       help='Path to model output CSV file')
    parser.add_argument('--base_data_path', type=str, default='data/data/',
                       help='Base path for dataset files')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Get script directory for resolving relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Resolve paths
    if not os.path.isabs(args.model_output_file):
        model_output_file = os.path.join(parent_dir, args.model_output_file)
    else:
        model_output_file = args.model_output_file
    
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(parent_dir, args.output_dir)
    else:
        output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = RuleBasedEvaluator(args.dataset, args.base_data_path)
    
    # Evaluate
    print(f"\nEvaluating: {model_output_file}")
    results_df, metrics = evaluator.evaluate_csv(model_output_file)
    
    # Save results
    base_filename = os.path.splitext(os.path.basename(model_output_file))[0]
    output_csv = os.path.join(output_dir, f"{base_filename}_rule_eval.csv")
    
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, f"{base_filename}_rule_eval_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("EVALUATION METRICS (RULE-BASED)\n")
        f.write("="*60 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"Metrics saved to: {metrics_file}")
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()


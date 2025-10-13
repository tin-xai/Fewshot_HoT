#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation using API Agents (Gemini/OpenAI/Claude)
=================================================================

This module provides evaluation of model outputs using various LLMs as judges.
It supports multi-threaded evaluation for faster processing.

Usage:
    python llm_judge.py --dataset date --model_output_file results/date/cot/fs_inst_gemini-2.0-flash-001_temp_10_full.csv --judge_model gemini-2.0-flash-001 --threads 8

Author: AI Assistant
"""

import os
import sys
import re
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agents.api_agents import api_agent
from data.load_dataset import DatasetLoader


@dataclass
class EvaluationExample:
    """Data class for holding evaluation examples."""
    question: str
    model_answer: str
    ground_truth: str
    example_id: int


@dataclass
class JudgmentResult:
    """Data class for holding judgment results."""
    example_id: int
    is_correct: bool
    judge_reasoning: str
    confidence: float
    error: Optional[str] = None


class PromptBuilder:
    """Build evaluation prompts for different types of tasks."""
    
    JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for reasoning tasks. Your job is to determine if a model's answer is correct by comparing it to the ground truth.

Question: {question}

Model Answer: {model_answer}

Ground Truth: {ground_truth}

Instructions:
Check if the final answer is correct or not.
Look for the final answer in curly brackets {{}} in the model answer if present.

Provide your evaluation in this exact format:

Reasoning: [Your detailed analysis of why the answer is correct or incorrect]

Final Judgment: [CORRECT or INCORRECT]

Confidence: [A number between 0.0 and 1.0 indicating your confidence]"""

    @classmethod
    def build_judge_prompt(cls, question: str, model_answer: str, ground_truth: str) -> str:
        """Build a prompt for the LLM judge."""
        return cls.JUDGE_PROMPT_TEMPLATE.format(
            question=question.strip(),
            model_answer=model_answer.strip(),
            ground_truth=str(ground_truth).strip()
        )


class ResultParser:
    """Parse and extract structured information from judge responses."""
    
    @staticmethod
    def parse_judgment(response: str) -> Tuple[bool, str, float]:
        """
        Parse the judgment response to extract decision, reasoning, and confidence.
        
        Returns:
            Tuple of (is_correct, reasoning, confidence)
        """
        if not response:
            return False, "No response from judge", 0.0
        
        # Extract reasoning
        reasoning_match = re.search(r"Reasoning:\s*(.*?)(?=Final Judgment:|$)", response, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        # Extract final judgment
        judgment_match = re.search(r"Final Judgment:\s*(CORRECT|INCORRECT)", response, re.IGNORECASE)
        is_correct = False
        if judgment_match:
            judgment = judgment_match.group(1).upper()
            is_correct = judgment == "CORRECT"
        else:
            # Fallback: look for correct/incorrect anywhere in the response
            if "CORRECT" in response.upper() and "INCORRECT" not in response.upper():
                is_correct = True
            elif "INCORRECT" in response.upper():
                is_correct = False
        
        # Extract confidence
        confidence_match = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        return is_correct, reasoning, confidence


class LLMJudgeEvaluator:
    """Main evaluator class that orchestrates the evaluation process."""
    
    def __init__(self, 
                 judge_model: str = "gemini-2.0-flash-001",
                 temperature: float = 0.0,
                 num_threads: int = 4,
                 verbose: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            judge_model: Name of the model to use as judge (e.g., 'gemini-2.0-flash-001', 'gpt-4', 'claude')
            temperature: Temperature for judge model (default 0.0 for deterministic results)
            num_threads: Number of parallel threads for evaluation
            verbose: Whether to print detailed progress
        """
        self.judge_model = judge_model
        self.temperature = temperature
        self.num_threads = num_threads
        self.verbose = verbose
        self.prompt_builder = PromptBuilder()
        self.parser = ResultParser()
        
        # Thread-safe progress tracking
        self.completed_count = 0
        self.lock = threading.Lock()
    
    def _evaluate_single_example(self, example: EvaluationExample) -> JudgmentResult:
        """Evaluate a single example using the LLM judge."""
        try:
            # Build prompt
            prompt = self.prompt_builder.build_judge_prompt(
                example.question, 
                example.model_answer, 
                example.ground_truth
            )
            
            # Get judgment from LLM
            response = api_agent(
                self.judge_model, 
                prompt, 
                temperature=self.temperature
            )
            
            if response is None:
                return JudgmentResult(
                    example_id=example.example_id,
                    is_correct=False,
                    judge_reasoning="Failed to get response from judge",
                    confidence=0.0,
                    error="API call failed"
                )
            
            # Parse the response
            is_correct, reasoning, confidence = self.parser.parse_judgment(response)
            
            # Update progress
            with self.lock:
                self.completed_count += 1
                if self.verbose and self.completed_count % 10 == 0:
                    print(f"Completed {self.completed_count} evaluations...")
            
            return JudgmentResult(
                example_id=example.example_id,
                is_correct=is_correct,
                judge_reasoning=reasoning,
                confidence=confidence
            )
            
        except Exception as e:
            return JudgmentResult(
                example_id=example.example_id,
                is_correct=False,
                judge_reasoning=f"Error during evaluation: {str(e)}",
                confidence=0.0,
                error=str(e)
            )
    
    def evaluate_batch(self, examples: List[EvaluationExample]) -> List[JudgmentResult]:
        """
        Evaluate a batch of examples using multi-threading.
        
        Args:
            examples: List of evaluation examples
            
        Returns:
            List of judgment results
        """
        results = []
        
        if self.verbose:
            print(f"\nEvaluating {len(examples)} examples using {self.num_threads} threads...")
            print(f"Judge model: {self.judge_model}")
            print(f"Temperature: {self.temperature}\n")
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks
            future_to_example = {
                executor.submit(self._evaluate_single_example, example): example
                for example in examples
            }
            
            # Collect results with progress bar
            with tqdm(total=len(examples), desc="Judging", disable=not self.verbose) as pbar:
                for future in as_completed(future_to_example):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
        
        # Sort results by example_id to maintain order
        results.sort(key=lambda x: x.example_id)
        
        return results


class DataLoader:
    """Load model outputs and ground truth data."""
    
    def __init__(self, dataset: str, base_data_path: str = 'data'):
        self.dataset = dataset
        self.base_data_path = base_data_path
        
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
    
    def load_model_outputs(self, csv_path: str) -> pd.DataFrame:
        """Load model outputs from CSV file."""
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} model outputs from {csv_path}")
        return df
    
    def load_ground_truth(self, ids: List[int]) -> List[Any]:
        """Load ground truth answers for given IDs."""
        ground_truths = self.dataset_loader.retrieve_gts(ids)
        print(f"Loaded {len(ground_truths)} ground truth answers")
        return ground_truths
    
    def prepare_evaluation_examples(self, csv_path: str) -> List[EvaluationExample]:
        """Prepare evaluation examples from CSV file."""
        df = self.load_model_outputs(csv_path)
        
        # Get ground truths
        ids = df['id'].tolist()
        ground_truths = self.load_ground_truth(ids)
        
        # Create evaluation examples
        examples = []
        for idx, row in df.iterrows():
            example = EvaluationExample(
                question=row['question'],
                model_answer=row['answer'],
                ground_truth=ground_truths[idx],
                example_id=row['id']
            )
            examples.append(example)
        
        return examples


class ResultExporter:
    """Export evaluation results to various formats."""
    
    @staticmethod
    def export_to_csv(results: List[JudgmentResult], output_path: str):
        """Export results to CSV file."""
        df = pd.DataFrame([
            {
                'example_id': r.example_id,
                'is_correct': r.is_correct,
                'confidence': r.confidence,
                'judge_reasoning': r.judge_reasoning,
                'error': r.error
            }
            for r in results
        ])
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    @staticmethod
    def export_detailed_results(
        results: List[JudgmentResult], 
        examples: List[EvaluationExample],
        output_path: str
    ):
        """Export detailed results with questions and answers."""
        data = []
        for result, example in zip(results, examples):
            data.append({
                'example_id': result.example_id,
                'question': example.question,
                'model_answer': example.model_answer,
                'ground_truth': example.ground_truth,
                'is_correct': result.is_correct,
                'confidence': result.confidence,
                'judge_reasoning': result.judge_reasoning,
                'error': result.error
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Detailed results saved to: {output_path}")
    
    @staticmethod
    def print_metrics(results: List[JudgmentResult]):
        """Print evaluation metrics."""
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        errors = sum(1 for r in results if r.error)
        
        accuracy = (correct / total * 100) if total > 0 else 0
        avg_confidence = sum(r.confidence for r in results) / total if total > 0 else 0
        
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Total Examples:     {total}")
        print(f"Correct:            {correct}")
        print(f"Incorrect:          {total - correct}")
        print(f"Errors:             {errors}")
        print(f"Accuracy:           {accuracy:.2f}%")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation")
    parser.add_argument('--dataset', type=str, default='date', 
                       help='Dataset name (e.g., date, GSM8K, etc.)')
    parser.add_argument('--model_output_file', type=str, required=True,
                       help='Path to model output CSV file')
    parser.add_argument('--judge_model', type=str, default='gemini-2.0-flash-001',
                       help='Judge model name (e.g., gemini-2.0-flash-001, gpt-4, claude)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for judge model (default: 0.0 for deterministic results)')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of parallel threads')
    parser.add_argument('--base_data_path', type=str, default='data/data/',
                       help='Base path for dataset files')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress')
    
    args = parser.parse_args()
    
    # Get script directory for resolving relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Resolve paths relative to parent directory if not absolute
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
    
    # Load data
    print(f"\nLoading data from: {model_output_file}")
    data_loader = DataLoader(args.dataset, args.base_data_path)
    examples = data_loader.prepare_evaluation_examples(model_output_file)
    
    # Initialize evaluator
    evaluator = LLMJudgeEvaluator(
        judge_model=args.judge_model,
        temperature=args.temperature,
        num_threads=args.threads,
        verbose=args.verbose
    )
    
    # Evaluate
    results = evaluator.evaluate_batch(examples)
    
    # Export results
    exporter = ResultExporter()
    
    # Generate output filenames
    base_filename = os.path.splitext(os.path.basename(model_output_file))[0]
    results_csv = os.path.join(output_dir, f"{base_filename}_judge_results.csv")
    detailed_csv = os.path.join(output_dir, f"{base_filename}_judge_detailed.csv")
    
    exporter.export_to_csv(results, results_csv)
    exporter.export_detailed_results(results, examples, detailed_csv)
    exporter.print_metrics(results)
    
    print(f"\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()


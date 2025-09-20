#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation using Ollama Llama 8B with Threading
================================================================

This module provides fast, multi-threaded evaluation of model outputs using
Llama 8B via Ollama as the judge. It supports various datasets and reasoning tasks.

Usage:
    python eval_llm_judge_ollama.py --dataset GSM8K --model_output_file results.csv --threads 8

Author: AI Assistant
"""

import os
import sys
import json
import re
import time
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from queue import Queue
import pandas as pd
import requests
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify that Ollama is running and the model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                if self.model_name not in model_names:
                    print(f"Warning: Model {self.model_name} not found. Available models: {model_names}")
                    print(f"To install, run: ollama pull {self.model_name}")
            else:
                raise ConnectionError(f"Ollama server returned status {response.status_code}")
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}. Please ensure Ollama is running.") from e
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> Optional[str]:
        """Generate response from Ollama model."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        except requests.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return None


class PromptBuilder:
    """Build evaluation prompts for different types of tasks."""
    
    JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for reasoning tasks. Your job is to determine if a model's answer is correct by comparing it to the ground truth.

Question: {question}

Model Answer: {model_answer}

Ground Truth: {ground_truth}

Instructions:
1. Carefully analyze the model's reasoning process and final answer
2. Compare it against the ground truth answer
3. Consider if the final answer is mathematically/logically equivalent even if the format differs
4. For math problems, focus on the numerical answer rather than exact wording
5. For multiple choice, check if the selected option is correct
6. For yes/no questions, check if the final stance matches

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
                 model_name: str = "llama3.1:8b",
                 ollama_url: str = "http://localhost:11434",
                 num_threads: int = 4,
                 verbose: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            model_name: Name of the Ollama model to use
            ollama_url: URL of the Ollama server
            num_threads: Number of parallel threads for evaluation
            verbose: Whether to print detailed progress
        """
        self.client = OllamaClient(model_name, ollama_url)
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
            response = self.client.generate(prompt, temperature=0.1, max_tokens=512)
            
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
        Evaluate a batch of examples using multithreading.
        
        Args:
            examples: List of evaluation examples
            
        Returns:
            List of judgment results
        """
        if self.verbose:
            print(f"Starting evaluation of {len(examples)} examples using {self.num_threads} threads...")
        
        self.completed_count = 0
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks
            future_to_example = {
                executor.submit(self._evaluate_single_example, example): example 
                for example in examples
            }
            
            # Collect results with progress bar
            if self.verbose:
                futures = tqdm(as_completed(future_to_example), total=len(examples), desc="Evaluating")
            else:
                futures = as_completed(future_to_example)
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    example = future_to_example[future]
                    results.append(JudgmentResult(
                        example_id=example.example_id,
                        is_correct=False,
                        judge_reasoning=f"Unexpected error: {str(e)}",
                        confidence=0.0,
                        error=str(e)
                    ))
        
        # Sort results by example_id to maintain order
        results.sort(key=lambda x: x.example_id)
        
        if self.verbose:
            print(f"Evaluation completed!")
        
        return results


class DatasetHandler:
    """Handle loading and processing of different datasets."""
    
    @staticmethod
    def load_model_outputs(file_path: str) -> List[Dict[str, Any]]:
        """Load model outputs from various file formats."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model output file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        elif file_ext == '.jsonl':
            results = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    results.append(json.loads(line.strip()))
            return results
        elif file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    @staticmethod
    def prepare_evaluation_examples(
        model_outputs: List[Dict[str, Any]], 
        dataset_name: str,
        config_path: str = '../configs/config.yaml',
        data_path: str = '../data/data/'
    ) -> List[EvaluationExample]:
        """
        Prepare evaluation examples by combining model outputs with ground truth.
        
        Args:
            model_outputs: List of model output records
            dataset_name: Name of the dataset
            config_path: Path to config file
            data_path: Base path to data directory
            
        Returns:
            List of evaluation examples
        """
        # Load dataset
        try:
            dataloader = DatasetLoader(
                config_path=config_path,
                base_data_path=data_path,
                base_few_shot_prompt_path='../fewshot_prompts/',
                dataset=dataset_name,
                data_mode='full',
                num_samples=None
            )
        except Exception as e:
            print(f"Warning: Could not load dataset {dataset_name} via DatasetLoader: {e}")
            print("Attempting to load directly from file...")
            return DatasetHandler._load_direct_from_file(model_outputs, dataset_name, data_path)
        
        examples = []
        model_outputs = model_outputs[:10]
        from tqdm import tqdm
        for i, output in enumerate(tqdm(model_outputs)):
            try:
                # Extract fields from model output
                question = output.get('question', '')
                model_answer = output.get('answer', '')
                example_id = output.get('id', i)
                
                # Get ground truth from dataset
                ground_truth = dataloader.retrieve_gts([example_id])
                if ground_truth:
                    gt = ground_truth[0]
                else:
                    print(f"Warning: No ground truth found for ID {example_id}")
                    continue
                
                examples.append(EvaluationExample(
                    question=question,
                    model_answer=model_answer,
                    ground_truth=gt,
                    example_id=example_id
                ))
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
        
        return examples
    
    @staticmethod
    def _load_direct_from_file(model_outputs: List[Dict[str, Any]], dataset_name: str, data_path: str) -> List[EvaluationExample]:
        """Fallback method to load ground truth directly from dataset files."""
        dataset_file = os.path.join(data_path, dataset_name, 'test.json')
        
        if not os.path.exists(dataset_file):
            print(f"Warning: Dataset file not found: {dataset_file}")
            print("Using model answers as-is without ground truth validation")
            
            examples = []
            for i, output in enumerate(model_outputs):
                examples.append(EvaluationExample(
                    question=output.get('question', ''),
                    model_answer=output.get('answer', ''),
                    ground_truth=output.get('ground_truth', 'N/A'),
                    example_id=output.get('id', i)
                ))
            return examples
        
        # Load ground truth
        with open(dataset_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        # Create mapping from ID to ground truth
        gt_map = {}
        for item in gt_data:
            gt_map[item.get('id', item.get('idx'))] = item.get('answer', item.get('target'))
        
        examples = []
        for i, output in enumerate(model_outputs):
            example_id = output.get('id', i)
            ground_truth = gt_map.get(example_id, 'N/A')
            
            examples.append(EvaluationExample(
                question=output.get('question', ''),
                model_answer=output.get('answer', ''),
                ground_truth=ground_truth,
                example_id=example_id
            ))
        
        return examples


def calculate_metrics(results: List[JudgmentResult]) -> Dict[str, float]:
    """Calculate evaluation metrics from judgment results."""
    if not results:
        return {}
    
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    errors = sum(1 for r in results if r.error is not None)
    
    # Calculate confidence-weighted accuracy
    total_confidence = sum(r.confidence for r in results)
    weighted_correct = sum(r.confidence for r in results if r.is_correct)
    
    metrics = {
        'accuracy': correct / total,
        'total_examples': total,
        'correct_examples': correct,
        'error_rate': errors / total,
        'average_confidence': total_confidence / total if total > 0 else 0.0,
        'confidence_weighted_accuracy': weighted_correct / total_confidence if total_confidence > 0 else 0.0
    }
    
    return metrics


def save_results(results: List[JudgmentResult], output_path: str, metrics: Dict[str, float]):
    """Save evaluation results to file."""
    # Prepare data for saving
    results_data = []
    for result in results:
        results_data.append({
            'example_id': result.example_id,
            'is_correct': result.is_correct,
            'judge_reasoning': result.judge_reasoning,
            'confidence': result.confidence,
            'error': result.error
        })
    
    # Save detailed results
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON with metrics
    final_output = {
        'metrics': metrics,
        'results': results_data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    # Also save a summary CSV
    csv_path = output_path.replace('.json', '_summary.csv')
    df = pd.DataFrame(results_data)
    df.to_csv(csv_path, index=False)
    
    print(f"Results saved to: {output_path}")
    print(f"Summary CSV saved to: {csv_path}")


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation using Ollama")
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., GSM8K, StrategyQA)')
    parser.add_argument('--model_output_file', type=str, required=True,
                       help='Path to file containing model outputs (CSV, JSON, or JSONL)')
    parser.add_argument('--output_file', type=str,
                       help='Path to save evaluation results (default: auto-generated)')
    parser.add_argument('--judge_model', type=str, default='qwen2.5:1.5b',
                       help='Ollama model to use as judge (default: llama3.1:8b)')
    parser.add_argument('--ollama_url', type=str, default='http://localhost:11434',
                       help='Ollama server URL (default: http://localhost:11434)')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of parallel threads (default: 4)')
    parser.add_argument('--config_path', type=str, default='../configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, default='../data/data/',
                       help='Base path to data directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Generate output file name if not provided
    if not args.output_file:
        base_name = os.path.splitext(os.path.basename(args.model_output_file))[0]
        args.output_file = f"{args.dataset}_{base_name}_llm_judge_results.json"
    
    try:
        print(f"Starting LLM-as-a-Judge evaluation...")
        print(f"Dataset: {args.dataset}")
        print(f"Model outputs: {args.model_output_file}")
        print(f"Judge model: {args.judge_model}")
        print(f"Threads: {args.threads}")
        print(f"Output: {args.output_file}")
        print("-" * 50)
        
        # Load model outputs
        print("Loading model outputs...")
        model_outputs = DatasetHandler.load_model_outputs(args.model_output_file)
        print(f"Loaded {len(model_outputs)} model outputs")
        
        # Prepare evaluation examples
        print("Preparing evaluation examples...")
        examples = DatasetHandler.prepare_evaluation_examples(
            model_outputs, args.dataset, args.config_path, args.data_path
        )
        print(f"Prepared {len(examples)} evaluation examples")
        
        if not examples:
            print("No valid examples found for evaluation!")
            return
        
        # Initialize evaluator
        evaluator = LLMJudgeEvaluator(
            model_name=args.judge_model,
            ollama_url=args.ollama_url,
            num_threads=args.threads,
            verbose=args.verbose
        )
        
        # Run evaluation
        start_time = time.time()
        results = evaluator.evaluate_batch(examples)
        end_time = time.time()
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        # Print results
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Total examples: {metrics['total_examples']}")
        print(f"Correct examples: {metrics['correct_examples']}")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Average confidence: {metrics['average_confidence']:.4f}")
        print(f"Confidence-weighted accuracy: {metrics['confidence_weighted_accuracy']:.4f}")
        print(f"Error rate: {metrics['error_rate']:.4f}")
        print(f"Evaluation time: {end_time - start_time:.2f} seconds")
        print(f"Average time per example: {(end_time - start_time) / len(examples):.2f} seconds")
        
        # Save results
        save_results(results, args.output_file, metrics)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    """
    ollama serve
    curl -s http://localhost:11434/api/tags

    python eval_llm_judge_ollama.py --dataset GSM8K --model_output_file results.csv --judge_model llama3:latest --threads 8
    """
    main()

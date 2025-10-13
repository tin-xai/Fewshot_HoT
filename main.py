"""
Main entry point for the LLM evaluation system.
Handles argument parsing, configuration setup, and orchestrates the evaluation process.
"""

import argparse
import sys
import os
from typing import List, Tuple

from configs.settings import Config
from data.dataset_handler import DatasetManager
from strategies.strategy_factory import StrategyFactory
from prompts.prompt_builder import PromptBuilder


def get_common_args() -> argparse.ArgumentParser:
    """Create and configure the argument parser with all available options."""
    arg_parser = argparse.ArgumentParser(
        description="LLM Evaluation System for various reasoning tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dataset commonsenseQA --answer_mode hot --temperature 0.7 --n_runs 3
  python main.py --dataset medQA --answer_mode ltm --num_samples 100 --save_answer
  python main.py --dataset date --answer_mode cot --n_runs 1 --num_samples 1 --save_answer --max_threads 8 --llm_model gemini-2.0-flash-001
        """
    )
    
    # Model Configuration
    arg_parser.add_argument(
        '--llm_model', 
        type=str, 
        default='gemini-1.5-flash-002', 
        help='The language model to query',
        choices=[
            'gemini-2.0-flash-001', 'claude', 
            'gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18', 
            'qwen25_coder_32b', 'qwq_32b', 'deepseek_r1',
            'nebius_llama70b', 'nebius_llama405b'
        ]
    )
    
    arg_parser.add_argument(
        '--temperature', 
        type=float, 
        default=1.0, 
        help='The temperature to use for the language model (0.0-2.0)'
    )
    
    # Dataset Configuration
    arg_parser.add_argument(
        '--dataset', 
        type=str, 
        default='GSM8K', 
        help='The dataset to evaluate on'
    )
    
    arg_parser.add_argument(
        '--data_mode', 
        type=str, 
        default='full', 
        help='How to select data samples',
        choices=['full', 'random', 'longest', 'shortest', 'remain']
    )
    
    arg_parser.add_argument(
        '--num_samples', 
        type=int, 
        default=200, 
        help='The number of samples to query (ignored if data_mode=full)'
    )
    
    # Prompting Strategy
    arg_parser.add_argument(
        '--prompt_used', 
        type=str, 
        default='fs_inst', 
        help='The prompting strategy to use',
        choices=['zs', 'fs', 'fs_inst']
    )
    
    arg_parser.add_argument(
        '--answer_mode', 
        type=str, 
        default='da', 
        help='The reasoning mode for generating answers',
        choices=['da', 'cot', 'hot', 'ltm', 'ltm_cot', 'ltm_hot', 'tot', 'cove', 'self_refine', 'cove_hot']
    )
    
    # File Paths
    arg_parser.add_argument(
        '--base_data_path', 
        type=str, 
        default='data/data/', 
        help='The base directory for dataset files'
    )
    
    arg_parser.add_argument(
        '--base_prompt_path', 
        type=str, 
        default='fewshot_prompts', 
        help='The base directory for few-shot prompt files'
    )
    
    arg_parser.add_argument(
        '--base_result_path', 
        type=str, 
        default='results', 
        help='The base directory for result files'
    )
    
    # Execution Options
    arg_parser.add_argument(
        '--n_runs', 
        type=int, 
        default=1, 
        help='The number of independent runs to execute'
    )
    
    arg_parser.add_argument(
        '--batch_request', 
        action='store_true', 
        help='Use batch API requests (currently supports GPT models only)'
    )
    
    arg_parser.add_argument(
        '--save_answer', 
        action='store_true', 
        help='Save individual answers and questions to CSV files'
    )
    
    arg_parser.add_argument(
        '--tail', 
        type=str, 
        default='', 
        choices=['', '_only_ground_Q', '_only_ground_A', '_repeat_Q', '_random_tag_Q'],
        help='The tail to use for the experiment'
    )
    
    # Debug and Development Options
    arg_parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug mode with additional logging'
    )
    
    arg_parser.add_argument(
        '--dry_run', 
        action='store_true', 
        help='Perform a dry run without making actual API calls'
    )
    
    arg_parser.add_argument(
        '--max_samples_debug', 
        type=int, 
        default=2, 
        help='Maximum samples to process in debug mode'
    )
    
    arg_parser.add_argument(
        '--max_threads', 
        type=int, 
        default=4, 
        help='Maximum number of threads for parallel processing (default: 4)'
    )
    
    return arg_parser

def print_experiment_info(config: Config):
    """Print information about the current experiment setup."""
    print("=" * 60)
    print("LLM EVALUATION EXPERIMENT")
    print("=" * 60)
    print(config)
    print("=" * 60)


def run_experiment(config: Config) -> bool:
    """Run the main evaluation experiment."""
    
    # Setup directories
    config.ensure_directories()
    
    # Initialize dataset manager
    print("Loading dataset...")
    dataset_manager = DatasetManager(config)
    questions, ids = dataset_manager.load_data()
    
    if not questions:
        print("Error: No questions loaded from dataset")
        return False
    
    print(f"Loaded {len(questions)} questions")
    
    # Initialize strategy
    print("Initializing evaluation strategy...")
    strategy = StrategyFactory.create_strategy(config)
    
    if not strategy:
        print("Error: Failed to create evaluation strategy")
        return False
    
    # Load few-shot prompts if needed
    few_shot_prompt = ""
    if config.needs_few_shot_prompt():
        print("Loading few-shot prompts...")
        few_shot_prompt = dataset_manager.load_few_shot_prompt(config.args.answer_mode)
    
    # Run experiments
    print(f"Starting evaluation with {config.args.n_runs} run(s)...")
    
    for run in range(1, config.args.n_runs + 1):
        print(f"\n--- Run {run}/{config.args.n_runs} ---")
        
        save_path = config.get_save_path_for_run(run)
        print(f"Results will be saved to: {save_path}")
        
        success = strategy.execute(questions, ids, few_shot_prompt, run, tail=config.args.tail)
        
        if success:
            print(f"Run {run} completed successfully")
        else:
            print(f"Run {run} failed")
            return False
    
    print("\nAll runs completed successfully!")
    return True


def main():
    """Main entry point for the application."""
    # Parse arguments
    arg_parser = get_common_args()
    args = arg_parser.parse_args()
    
    # Initialize configuration
    config = Config(args)
    
    # Print experiment information
    print_experiment_info(config)
    
    # Handle dry run mode
    if args.dry_run:
        print("\nDRY RUN MODE - No actual API calls will be made")
        print("Configuration validated successfully")
        return
    
    # Run the experiment
    success = run_experiment(config)
    
    if success:
        print("\nExperiment completed successfully!")
        sys.exit(0)
    else:
        print("\nExperiment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
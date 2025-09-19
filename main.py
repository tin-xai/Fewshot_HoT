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


def get_common_args() -> argparse.ArgumentParser:
    """Create and configure the argument parser with all available options."""
    arg_parser = argparse.ArgumentParser(
        description="LLM Evaluation System for various reasoning tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dataset GSM8K --answer_mode cot --llm_model gpt-4o-2024-08-06
  python main.py --dataset commonsenseQA --answer_mode hot --temperature 0.7 --n_runs 3
  python main.py --dataset medQA --answer_mode ltm --num_samples 100 --save_answer
        """
    )
    
    # Model Configuration
    arg_parser.add_argument(
        '--llm_model', 
        type=str, 
        default='gemini-1.5-pro-002', 
        help='The language model to query',
        choices=[
            'gemini-1.5-pro-002', 'gemini-1.5-flash-002', 'claude', 
            'gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18', 
            'llama_transformer', 'llama_groq', 'llama_together',
            'llama_sambanova_70b', 'llama_sambanova_8b', 'llama_sambanova_405b',
            'qwen25_coder_32b', 'qwq_32b', 'deepseek_r1', 'gemini_thinking',
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
        default='longest', 
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
        default='data', 
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
    
    return arg_parser


def validate_args(args) -> bool:
    """Validate argument combinations and values."""
    errors = []
    
    # Validate temperature range
    if not (0.0 <= args.temperature <= 2.0):
        errors.append("Temperature must be between 0.0 and 2.0")
    
    # Validate batch request compatibility
    if args.batch_request and 'gpt' not in args.llm_model:
        errors.append("Batch requests are currently only supported for GPT models")
    
    # Validate multi-step modes
    multi_step_modes = ['ltm', 'ltm_hot', 'tot', 'cove', 'cove_hot', 'self_refine']
    if args.answer_mode in multi_step_modes and args.prompt_used != 'fs_inst':
        print(f"Warning: Multi-step mode '{args.answer_mode}' typically works best with 'fs_inst' prompting")
    
    # Validate file paths exist
    if not os.path.exists(args.base_data_path):
        errors.append(f"Base data path does not exist: {args.base_data_path}")
    
    if errors:
        print("Argument validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def print_experiment_info(config: Config):
    """Print information about the current experiment setup."""
    print("=" * 60)
    print("LLM EVALUATION EXPERIMENT")
    print("=" * 60)
    print(config)
    print("=" * 60)


def run_experiment(config: Config) -> bool:
    """Run the main evaluation experiment."""
    try:
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
            
            success = strategy.execute(questions, ids, few_shot_prompt, run, save_path)
            
            if success:
                print(f"Run {run} completed successfully")
            else:
                print(f"Run {run} failed")
                return False
        
        print("\nAll runs completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during experiment execution: {str(e)}")
        if config.args.debug:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main entry point for the application."""
    # Parse arguments
    arg_parser = get_common_args()
    args = arg_parser.parse_args()
    
    # Validate arguments
    if not validate_args(args):
        sys.exit(1)
    
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
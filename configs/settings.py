"""Configuration management for the LLM evaluation system."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Main configuration class that handles all settings and path management."""
    
    def __init__(self, args=None):
        """Initialize configuration with parsed arguments."""
        self.args = args
        self._setup_paths()
        self._setup_experiment_settings()
    
    def _setup_paths(self):
        """Setup all file paths based on experiment configuration."""
        self.save_result_folder = self._determine_result_folder()
        self.save_path = self._build_save_path()
        self.base_data_path = self.args.base_data_path
        self.base_prompt_path = 'fewshot_prompts/'
        self.config_path = 'configs/config.yaml'
    
    def _determine_result_folder(self) -> str:
        """Determine the appropriate result folder based on answer mode."""
        answer_mode = self.args.answer_mode
        
        if answer_mode in ['ltm', 'ltm_hot']:
            return 'results_LtM'
        elif answer_mode == 'tot':
            return 'results_tot'
        elif answer_mode == 'cove':
            return 'results_cove'
        elif answer_mode == 'cove_hot':
            return 'results_cove_hot'
        elif answer_mode == 'self_refine':
            return 'results_self_refine'
        else:
            # Default folder for standard modes (da, cot, hot)
            return 'results_ifbench'
    
    def _build_save_path(self) -> str:
        """Build the complete save path for results."""
        # Create base save path
        base_path = f'{self.save_result_folder}/{self.args.dataset}/{self.args.answer_mode}/{self.args.prompt_used}_{self.args.llm_model}.csv'
        
        # Add temperature suffix
        str_temperature = str(self.args.temperature).replace('.', '')
        
        # Add additional suffixes based on configuration
        tail = self.args.tail
        save_path = base_path[:-4] + f'_temp_{str_temperature}_{self.args.data_mode}{tail}.csv'
        
        return save_path

    
    def _setup_experiment_settings(self):
        """Setup experiment-specific settings."""
        self.max_tokens = 8192
        self.max_retries = 3
        self.batch_output_file = f'batch_request/{self.args.dataset}/{self.args.answer_mode}/records.jsonl'
        self.batch_result_file = f'batch_request/{self.args.dataset}/{self.args.answer_mode}/{self.args.prompt_used}_{self.args.llm_model}.jsonl'
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            f'{self.save_result_folder}/{self.args.dataset}/{self.args.answer_mode}',
            f'batch_request/{self.args.dataset}',
            f'batch_request/{self.args.dataset}/{self.args.answer_mode}'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_save_path_for_run(self, run_number: int) -> str:
        """Get the save path for a specific run number."""
        if run_number == 1:
            return self.save_path
        
        # Handle multiple runs
        if '_run_' in self.save_path:
            return self.save_path.replace(f'_run_{run_number-1}', f'_run_{run_number}')
        else:
            return self.save_path[:-4] + f'_run_{run_number}.csv'
    
    def is_multi_step_mode(self) -> bool:
        """Check if the current answer mode requires multi-step processing."""
        return self.args.answer_mode in ['ltm', 'ltm_hot', 'tot', 'cove', 'cove_hot', 'self_refine']
    
    def needs_few_shot_prompt(self) -> bool:
        """Check if the current mode needs few-shot prompts."""
        return not self.is_multi_step_mode()
    
    def get_dataset_choices(self) -> list:
        """Get all available dataset choices."""
        return [
            'GSM8K', 'StrategyQA', 'p_GSM8K', 'AQUA', 'MultiArith', 'ASDiv', 'SVAMP',
            'commonsenseQA', 'wikimultihopQA', 'date', 'sports', 'reclor', 'CLUTRR',
            'object_counting', 'navigate', 'causal_judgement', 'logical_deduction_three_objects',
            'logical_deduction_five_objects', 'logical_deduction_seven_objects',
            'reasoning_about_colored_objects', 'GSM_Plus', 'GSM_IC', 'spartQA',
            'last_letter_2', 'last_letter_4', 'coin', 'word_sorting',
            'tracking_shuffled_objects_seven_objects', 'gpqa', 'GSM8K_Hard',
            'web_of_lies', 'temporal_sequences', 'drop_break', 'drop_cencus',
            'squad', 'medQA', 'GSM_Symbolic', 'LIMO', 'bbeh_causal_judgement',
            'bbeh_boardgame', 'bbeh_object_attribute', 'bbeh_spatial_reasoning'
        ]
    
    def get_llm_model_choices(self) -> list:
        """Get all available LLM model choices."""
        return [
            'gemini-1.5-pro-002', 'gemini-1.5-flash-002', 'claude', 'gpt-4o-2024-08-06',
            'gpt-4o-mini-2024-07-18', 'llama_transformer', 'llama_groq', 'llama_together',
            'qwen25_coder_32b', 'qwq_32b', 'deepseek_r1', 'gemini_thinking',
            'nebius_llama70b', 'nebius_llama405b'
        ]
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"""Configuration:
  Model: {self.args.llm_model}
  Dataset: {self.args.dataset}
  Answer Mode: {self.args.answer_mode}
  Prompt Used: {self.args.prompt_used}
  Temperature: {self.args.temperature}
  Data Mode: {self.args.data_mode}
  Num Samples: {self.args.num_samples}
  Num Runs: {self.args.n_runs}
  Save Path: {self.save_path}
  Batch Request: {self.args.batch_request}"""


class ExperimentConfig:
    """Additional configuration for experimental variations."""
    
    # Tail variations for different experimental conditions
    TAIL_VARIATIONS = {
        'default': '',
        'only_ground_Q': '_only_ground_Q',
        'only_ground_A': '_only_ground_A', 
        'repeat_Q': '_repeat_Q',
        'random_tag': '_random_tag_Q'
    }
    
    # Multi-step strategy specific configurations
    MULTI_STEP_CONFIGS = {
        'ltm': {
            'max_sub_questions': 3,
            'decomposition_required': True
        },
        'tot': {
            'num_paths': 3,
            'evaluation_required': True
        },
        'cove': {
            'num_verification_questions': 3,
            'revision_required': True
        },
        'self_refine': {
            'max_refinement_rounds': 2,
            'critique_required': True
        }
    }
    
    @classmethod
    def get_multi_step_config(cls, answer_mode: str) -> dict:
        """Get configuration for a specific multi-step strategy."""
        return cls.MULTI_STEP_CONFIGS.get(answer_mode, {})
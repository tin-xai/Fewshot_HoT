import pandas as pd
from data.load_dataset import DatasetLoader

class DatasetManager:
    def __init__(self, config):
        self.config = config
        self.dataloader = DatasetLoader(
            config_path='configs/config.yaml',
            base_data_path=config.args.base_data_path,
            base_few_shot_prompt_path='fewshot_prompts/',
            dataset=config.args.dataset,
            data_mode=config.args.data_mode,
            num_samples=config.args.num_samples
        )
    
    def load_data(self):
        questions, ids = self.dataloader.get_questions_and_ids()
        
        if self.config.args.data_mode == 'remain':
            return self._load_remaining_data()
        
        return questions, ids
    
    def load_few_shot_prompt(self, answer_mode):
        if answer_mode in ['ltm', 'ltm_hot', 'tot', 'cove', 'cove_hot', 'self_refine']:
            return ""
        
        return self.dataloader._load_few_shot_prompt(fs_mode=answer_mode)
        
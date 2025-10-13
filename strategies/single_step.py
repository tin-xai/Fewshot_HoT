from .base_strategy import BaseStrategy
from prompts.prompt_builder import PromptBuilder
from agents.api_agents import api_agent

class SingleStepStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.prompt_builder = PromptBuilder(config)
    
    
    def generate_response(self, question, dataset, few_shot_prompt, tail=""):
        prompt = self.prompt_builder.create_prompt(
            question, 
            dataset, 
            self.config.args.prompt_used,
            few_shot_prompt,
            self.config.args.answer_mode,
            tail
        )
        
        return api_agent(self.llm_model, prompt, temperature=self.temperature)
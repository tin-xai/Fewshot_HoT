from .base_strategy import BaseStrategy
from prompts.prompt_builder import PromptBuilder
from agents.api_agents import api_agent

class SingleStepStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.prompt_builder = PromptBuilder(config)
    
    def execute(self, questions, ids, run_number):
        answers = []
        for question, id in zip(questions, ids):
            response = self.generate_response(question, self.config.args.dataset)
            if response:
                answers.append(response)
        return answers
    
    def generate_response(self, question, dataset):
        prompt = self.prompt_builder.create_prompt(
            question, 
            dataset, 
            self.config.args.prompt_used,
            self.config.args.answer_mode
        )
        return api_agent(self.llm_model, prompt, temperature=self.temperature)
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, config):
        self.config = config
        self.llm_model = config.args.llm_model
        self.temperature = config.args.temperature
    
    @abstractmethod
    def execute(self, questions, ids, run_number):
        pass
    
    @abstractmethod
    def generate_response(self, question, dataset):
        pass
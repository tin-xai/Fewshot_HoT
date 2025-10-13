from .single_step import SingleStepStrategy
from .multi_step.tot import TreeOfThoughtStrategy
from .multi_step.ltm import LeastToMostStrategy
from .multi_step.cove import ChainOfVerificationStrategy, ChainOfVerificationHoTStrategy
from .multi_step.self_refine import SelfRefineStrategy

class StrategyFactory:
    @staticmethod
    def create_strategy(config):
        if config.args.answer_mode in ['cot', 'hot']:
            return SingleStepStrategy(config)
        elif config.args.answer_mode == 'tot':
            return TreeOfThoughtStrategy(config)
        elif config.args.answer_mode == 'ltm':
            return LeastToMostStrategy(config)
        elif config.args.answer_mode == 'cove':
            return ChainOfVerificationStrategy(config)
        elif config.args.answer_mode == 'self_refine':
            return SelfRefineStrategy(config)
        elif config.args.answer_mode == 'cove_hot':
            return ChainOfVerificationHoTStrategy(config)
        else:
            raise ValueError(f"Unknown answer mode: {config.args.answer_mode}")





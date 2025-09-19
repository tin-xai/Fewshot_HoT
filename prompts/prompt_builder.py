"""
Prompt builder for creating different types of prompts based on strategy and dataset.
Handles few-shot, zero-shot, and instructional prompting with various answer modes.
"""

from typing import Optional
from .prompt_utils import (
    extract_last_sentence, 
    remove_fact_tags_from_answers,
    remove_fact_tags_from_questions
)


class PromptBuilder:
    """
    Builds prompts for different prompting strategies and answer modes.
    
    Supports:
    - Few-shot (fs): Uses examples without additional instructions
    - Few-shot instructional (fs_inst): Uses examples with specific instructions
    - Zero-shot (zs): No examples, instruction-based only
    
    Answer modes:
    - da: Direct answer
    - cot: Chain-of-thought
    - hot: Highlighted Chain-of-thought (with fact tagging/grounding)
    """
    
    def __init__(self, config):
        self.config = config
        self.dataset = config.args.dataset
    
    def create_prompt(self, question: str, dataset: str, prompt_used: str, 
                     few_shot_prompt: str, answer_mode: str, tail: str = "") -> Optional[str]:
        """
        Create a prompt based on the specified strategy and answer mode.
        
        Args:
            question: The question to answer
            dataset: Dataset name for context-specific processing
            prompt_used: Prompting strategy ('fs', 'fs_inst', 'zs')
            few_shot_prompt: Few-shot examples (if applicable)
            answer_mode: Answer generation mode ('da', 'cot', 'hot', etc.)
            tail: Additional suffix for experimental variations
            
        Returns:
            Formatted prompt string or None for multi-step modes
        """
        # Multi-step modes are handled by their respective strategy classes
        if answer_mode in ['ltm', 'ltm_hot', 'tot', 'cove', 'self_refine', 'cove_hot']:
            return None
        
        if prompt_used == "fs":
            return self._build_few_shot_prompt(question, few_shot_prompt)
        elif prompt_used == "fs_inst":
            return self._build_instructional_prompt(question, dataset, few_shot_prompt, answer_mode, tail)
        elif prompt_used == "zs":
            return self._build_zero_shot_prompt(question, dataset, answer_mode)
        else:
            raise ValueError(f"Unknown prompt_used: {prompt_used}")
    
    def _build_few_shot_prompt(self, question: str, few_shot_prompt: str) -> str:
        """Build a simple few-shot prompt with examples."""
        return f"{few_shot_prompt}\n{question}"
    
    def _build_instructional_prompt(self, question: str, dataset: str, few_shot_prompt: str,
                                  answer_mode: str, tail: str) -> str:
        """Build few-shot prompt with specific instructions based on answer mode."""
        last_sentence = extract_last_sentence(question, dataset)
        
        if answer_mode == 'da':
            instruction = self._get_direct_answer_instruction()
        elif answer_mode == 'cot':
            instruction = self._get_chain_of_thought_instruction()
        elif answer_mode == 'hot':
            instruction = self._get_hot_of_thought_instruction(last_sentence, tail)
        else:
            raise ValueError(f"Unknown answer_mode for instructional prompt: {answer_mode}")
        
        # Apply tail-specific transformations to few-shot prompt if needed
        processed_few_shot = self._apply_tail_transformations(few_shot_prompt, tail)
        
        return f"{processed_few_shot}\n{question}\n{instruction}"
    
    def _build_zero_shot_prompt(self, question: str, dataset: str, answer_mode: str) -> str:
        """Build zero-shot prompt with instructions but no examples."""
        last_sentence = extract_last_sentence(question, dataset)
        
        if answer_mode == 'hot':
            instruction = f"""I want you to answer this question but your explanation should contain references referring back to the information in the question. To do that, first, re-generate the question with proper tags (<fact1>, <fact2>, <fact3>, etc) for key phrases, the key phrases that are most relevant to answering the question {last_sentence}, and then generate your answers that also have the tag (<fact1>, <fact2>, <fact3>, etc) for the grounded information. Give your answer by analyzing step by step, and give your answer in curly brackets """ + "{} in the final answer. The output format is as follow:\n\
            Reformatted Question: \
                Answer:\
                    Final answer:"
        else:
            # Default zero-shot instruction for other modes
            instruction = f"Please answer the following question step by step and provide your final answer in curly brackets {{}}."
        
        return f"{question}\n{instruction}"
    
    def _get_direct_answer_instruction(self) -> str:
        """Get instruction for direct answer mode."""
        return "Do not generate your explanation, please give the answer only as follow:\nAnswer:."
    
    def _get_chain_of_thought_instruction(self) -> str:
        """Get instruction for chain-of-thought mode."""
        return "Please generate your explanation first, then generate the final answer in the bracket as follow:\nAnswer: {}"
    
    def _get_hot_of_thought_instruction(self, last_sentence: str, tail: str) -> str:
        """Get instruction for hot-of-thought (grounding) mode with tail variations."""
        
        # Base HoT instruction (ground in both Q and A)
        if tail == 'derive':
            # Standard HoT instruction with derivation tags
            return f"""I want you to answer this question but your explanation should contain references referring back to the information in the question. To do that, first, re-generate the question with proper tags for key phrases, the key phrases that are most relevant to answering the question {last_sentence} and then generate your answers, in your answer, also tag the intermediate information in <derive1>, <derive2>, etc., reflecting how you deduced new information. And enclose the ultimate answer in a curly bracket {{}}. The output format is as follow:
                    Reformatted Question: 
                        Answer: """
        
        # Ground in Q only
        elif tail == '_only_ground_Q':
            return f"""I want you to answer this question. To do that, first, re-generate the question with proper tags for key phrases, the key phrases that are most relevant to answering the question {last_sentence}, and then generate your answers. And enclose the ultimate answer in a curly bracket {{}}. The output format is as follow:
                    Reformatted Question: 
                        Answer:"""
        
        # Ground in A only
        elif tail == '_only_ground_A':
            return f"""I want you to answer this question but your explanation should contain references referring back to the information in the question. To do that, first, repeat the question and then, generate your answers with proper tags for key phrases, the key phrases that are most relevant to answering the question {last_sentence}. The output format is as follow:
                Reformatted Question: 
                    Answer:"""
        
        # Repeat question without grounding
        elif tail == '_repeat_Q':
            return f"""I want you to answer this question. To do that, first, repeat the question and then, generate your answers. And enclose the ultimate answer in a curly bracket {{}}. The output format is as follow:
                Reformatted Question: 
                    Answer:"""
        
        else:
            # Fallback to standard HoT
            return f"""I want you to answer this question but your explanation should contain references referring back to the information in the question. To do that, first, re-generate the question with proper tags for key phrases, the key phrases that are most relevant to answering the question {last_sentence} and then generate your answers. The output format is as follow:
                    Reformatted Question: 
                        Answer: """
    
    def _apply_tail_transformations(self, few_shot_prompt: str, tail: str) -> str:
        """Apply tail-specific transformations to the few-shot prompt."""
        if tail == '_repeat_Q':
            # Remove fact tags from both questions and answers
            processed_prompt = remove_fact_tags_from_answers(few_shot_prompt)
            processed_prompt = remove_fact_tags_from_questions(processed_prompt)
            return processed_prompt
        
        elif tail == '_only_ground_Q':
            # Remove fact tags from answers only
            return remove_fact_tags_from_answers(few_shot_prompt)
        
        elif tail == '_only_ground_A':
            # Remove fact tags from questions and reformatted question lines
            processed_prompt = remove_fact_tags_from_questions(few_shot_prompt)
            # Also remove "Reformatted Question:" lines entirely
            import re
            processed_prompt = re.sub(r'Reformatted Question:.*?\n', '', processed_prompt)
            return processed_prompt
        
        # No transformation needed for other tails
        return few_shot_prompt


class AdvancedPromptBuilder(PromptBuilder):
    """
    Extended prompt builder with additional capabilities for experimental variations.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.experimental_modes = {
            'random_tag': self._apply_random_tag_variation,
            'mismatched_tag': self._apply_mismatched_tag_variation,
            'progressive_complexity': self._apply_progressive_complexity
        }
    
    def create_experimental_prompt(self, question: str, dataset: str, prompt_used: str,
                                 few_shot_prompt: str, answer_mode: str, 
                                 experimental_mode: str) -> Optional[str]:
        """Create experimental prompt variations."""
        base_prompt = self.create_prompt(question, dataset, prompt_used, 
                                       few_shot_prompt, answer_mode)
        
        if not base_prompt or experimental_mode not in self.experimental_modes:
            return base_prompt
        
        return self.experimental_modes[experimental_mode](base_prompt, question, dataset)
    
    def _apply_random_tag_variation(self, prompt: str, question: str, dataset: str) -> str:
        """Apply random tag variation for experimental control."""
        # This could randomize fact tags or apply meaningless tags
        # Implementation would depend on specific experimental needs
        return prompt
    
    def _apply_mismatched_tag_variation(self, prompt: str, question: str, dataset: str) -> str:
        """Apply mismatched tags for experimental analysis."""
        # This could intentionally mismatch tags between questions and answers
        return prompt
    
    def _apply_progressive_complexity(self, prompt: str, question: str, dataset: str) -> str:
        """Apply progressive complexity increases."""
        # This could add complexity layers to test model capabilities
        return prompt


class DatasetSpecificPromptBuilder(PromptBuilder):
    """
    Dataset-specific prompt builder that handles unique requirements for different datasets.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.dataset_handlers = {
            'commonsenseQA': self._handle_commonsense_qa,
            'medQA': self._handle_medical_qa,
            'GSM8K': self._handle_math_problems,
            'spartQA': self._handle_spartan_qa,
            'sports': self._handle_sports_plausibility,
            'reclor': self._handle_reading_comprehension,
        }
    
    def create_prompt(self, question: str, dataset: str, prompt_used: str,
                     few_shot_prompt: str, answer_mode: str, tail: str = "") -> Optional[str]:
        """Create dataset-specific prompt with special handling if needed."""
        # Apply dataset-specific preprocessing if handler exists
        if dataset in self.dataset_handlers:
            question = self.dataset_handlers[dataset](question)
        
        # Use parent class method for main prompt creation
        return super().create_prompt(question, dataset, prompt_used, 
                                   few_shot_prompt, answer_mode, tail)
    
    def _handle_commonsense_qa(self, question: str) -> str:
        """Handle CommonSenseQA specific formatting."""
        # Could add specific processing for multiple choice format
        return question
    
    def _handle_medical_qa(self, question: str) -> str:
        """Handle medical QA specific requirements."""
        # Could add medical context preprocessing
        return question
    
    def _handle_math_problems(self, question: str) -> str:
        """Handle math problem specific formatting."""
        # Could add mathematical notation processing
        return question
    
    def _handle_spartan_qa(self, question: str) -> str:
        """Handle SpartanQA specific requirements."""
        # SpartanQA might have empty last_sentence handling
        return question
    
    def _handle_sports_plausibility(self, question: str) -> str:
        """Handle sports plausibility questions."""
        # Could add sports-specific context
        return question
    
    def _handle_reading_comprehension(self, question: str) -> str:
        """Handle reading comprehension questions."""
        # Could add passage-specific processing
        return question


class PromptTemplate:
    """
    Template class for storing and managing prompt templates.
    """
    
    def __init__(self, name: str, template: str, placeholders: list):
        self.name = name
        self.template = template
        self.placeholders = placeholders
    
    def format(self, **kwargs) -> str:
        """Format the template with provided values."""
        missing_placeholders = set(self.placeholders) - set(kwargs.keys())
        if missing_placeholders:
            raise ValueError(f"Missing required placeholders: {missing_placeholders}")
        
        return self.template.format(**kwargs)


class PromptTemplateManager:
    """
    Manages a collection of prompt templates for different use cases.
    """
    
    def __init__(self):
        self.templates = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize commonly used prompt templates."""
        
        # Direct Answer Template
        self.add_template(
            "direct_answer",
            "{few_shot_prompt}\n{question}\nDo not generate your explanation, please give the answer only as follow:\nAnswer:.",
            ["few_shot_prompt", "question"]
        )
        
        # Chain-of-Thought Template
        self.add_template(
            "chain_of_thought",
            "{few_shot_prompt}\n{question}\nPlease generate your explanation first, then generate the final answer in the bracket as follow:\nAnswer: {{}}",
            ["few_shot_prompt", "question"]
        )
        
        # Hot-of-Thought Template
        self.add_template(
            "hot_of_thought",
            "{few_shot_prompt}\n{question}\n{hot_instruction}",
            ["few_shot_prompt", "question", "hot_instruction"]
        )
        
        # Zero-shot Template
        self.add_template(
            "zero_shot",
            "{question}\n{instruction}",
            ["question", "instruction"]
        )
    
    def add_template(self, name: str, template: str, placeholders: list):
        """Add a new template to the manager."""
        self.templates[name] = PromptTemplate(name, template, placeholders)
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def format_template(self, name: str, **kwargs) -> str:
        """Format a template with provided values."""
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        return template.format(**kwargs)
    
    def list_templates(self) -> list:
        """List all available template names."""
        return list(self.templates.keys())
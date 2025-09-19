from abc import ABC, abstractmethod
import pandas as pd
import os

class BaseStrategy(ABC):
    def __init__(self, config):
        self.config = config
        self.llm_model = config.args.llm_model
        self.temperature = config.args.temperature
        self.max_tokens = config.max_tokens
    
    def execute(self, questions, ids, few_shot_prompt, run_number, tail=""):
        """
        Default execute implementation that can be overridden by specific strategies.
        This handles both single-step and multi-step strategies.
        """
        answers = []
        save_path = self.config.get_save_path_for_run(run_number)
        
        for question, id in zip(questions, ids):
            response = self.generate_response(question, self.config.args.dataset, few_shot_prompt, tail)
            print(response)
            if response:
                answers.append(response)
                # Save response immediately on the fly
                if self.config.args.save_answer:
                    self.save_response(id, question, response, save_path)
        return answers
    
    @abstractmethod
    def generate_response(self, question, dataset, few_shot_prompt, tail=""):
        pass
    
    def save_response(self, question_id, question, answer, save_path):
        """
        Save a single response to CSV file on the fly.
        
        Args:
            question_id: ID of the question
            question: The question text
            answer: The generated answer
            save_path: Path to save the CSV file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(save_path)
        
        # Create a single row dataframe
        df_row = pd.DataFrame({
            'id': [question_id],
            'question': [question],
            'answer': [answer]
        })
        
        # Append to file (create if doesn't exist)
        if file_exists:
            # Read existing data and append new row
            try:
                existing_df = pd.read_csv(save_path)
                updated_df = pd.concat([existing_df, df_row], ignore_index=True)
                updated_df.to_csv(save_path, index=False)
            except Exception as e:
                print(f"Warning: Could not append to existing file {save_path}: {e}")
                # If reading fails, just write the new row with headers
                df_row.to_csv(save_path, index=False)
        else:
            # Create new file with headers
            df_row.to_csv(save_path, index=False)
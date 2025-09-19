from abc import ABC, abstractmethod
import pandas as pd
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class BaseStrategy(ABC):
    def __init__(self, config):
        self.config = config
        self.llm_model = config.args.llm_model
        self.temperature = config.args.temperature
        self.max_tokens = config.max_tokens
        self._save_lock = threading.Lock()  # Thread lock for safe file writing
    
    def execute(self, questions, ids, few_shot_prompt, run_number, tail=""):
        """
        Default execute implementation with multithreading support.
        This handles both single-step and multi-step strategies.
        """
        answers = []
        save_path = self.config.get_save_path_for_run(run_number)
        
        # Determine number of threads (limit to avoid overwhelming APIs)
        max_workers = min(len(questions), getattr(self.config.args, 'max_threads', 4))
        
        def process_single_question(question_data):
            """Process a single question and return result"""
            question, question_id = question_data
            try:
                response = self.generate_response(question, self.config.args.dataset, few_shot_prompt, tail)
                print(f"[Thread] Completed question {question_id}: {question[:50]}...")
                if response:
                    # Save response immediately on the fly (thread-safe)
                    if self.config.args.save_answer:
                        self.save_response(question_id, question, response, save_path)
                    return (question_id, response)
                return None
            except Exception as e:
                print(f"Error processing question {question_id}: {str(e)}")
                return None
        
        # Use ThreadPoolExecutor for parallel processing
        question_data = list(zip(questions, ids))
        completed_responses = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            print(f"Processing {len(questions)} questions with {max_workers} threads...")
            
            # Submit all tasks
            future_to_data = {
                executor.submit(process_single_question, data): data 
                for data in question_data
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_data):
                result = future.result()
                if result:
                    question_id, response = result
                    completed_responses.append((question_id, response))
        
        # Sort responses by original order and extract answers
        completed_responses.sort(key=lambda x: x[0])  # Sort by question_id
        answers = [response for _, response in completed_responses]
        
        print(f"Completed processing {len(answers)} out of {len(questions)} questions")
        return answers
    
    @abstractmethod
    def generate_response(self, question, dataset, few_shot_prompt, tail=""):
        pass
    
    def save_response(self, question_id, question, answer, save_path):
        """
        Save a single response to CSV file on the fly (thread-safe).
        
        Args:
            question_id: ID of the question
            question: The question text
            answer: The generated answer
            save_path: Path to save the CSV file
        """
        with self._save_lock:  # Thread-safe file writing
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
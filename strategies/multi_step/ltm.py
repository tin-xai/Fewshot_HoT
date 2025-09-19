"""
Least-to-Most (LtM) prompting strategy implementation.
Decomposes complex problems into simpler sub-problems and solves them iteratively.
"""

import re
from typing import List, Optional, Tuple

from ..base_strategy import BaseStrategy
from agents.api_agents import api_agent
from prompts.prompt_utils import extract_last_sentence


class LeastToMostStrategy(BaseStrategy):
    """
    Least-to-Most prompting strategy.
    
    Process:
    1. Decompose the main question into 2-3 simpler sub-questions
    2. Solve each sub-question iteratively, using previous answers as context
    3. Combine all sub-answers to generate the final answer
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.max_sub_questions = config.args.get('max_sub_questions', 3)
    
    def generate_response(self, question: str, dataset: str, few_shot_prompt: str = "", tail: str = "") -> Optional[str]:
        """Generate response using Least-to-Most prompting."""
        try:
            # Step 1: Decompose the question
            decomposition_response = self._decompose_question(question, dataset)
            if not decomposition_response:
                return None
            
            # Step 2: Extract sub-questions
            sub_questions = self._extract_sub_questions(decomposition_response)
            if not sub_questions:
                return None
            
            # Step 3: Solve each sub-question iteratively
            sub_answers = self._solve_sub_questions(sub_questions, question)
            if not sub_answers:
                return None
            
            # Step 4: Combine for final answer
            final_answer = self._combine_answers(question, sub_questions, sub_answers)
            
            # Compile full response
            full_response = self._compile_full_response(
                decomposition_response, sub_questions, sub_answers, final_answer
            )
            
            return full_response
            
        except Exception as e:
            print(f"Error in LeastToMostStrategy: {str(e)}")
            return None
    
    def _decompose_question(self, question: str, dataset: str) -> Optional[str]:
        """Decompose the main question into sub-questions."""
        last_sentence = extract_last_sentence(question, dataset)
        
        decompose_prompt = f"""{question}

To solve this question effectively, I need to break it down into simpler sub-questions. Please identify 2-3 key sub-questions that need to be answered to solve the main problem: {last_sentence}

Please list only the sub-questions in this format:
1. [First sub-question]
2. [Second sub-question]
3. [Third sub-question (if needed)]"""

        return api_agent(
            self.llm_model, 
            decompose_prompt, 
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _extract_sub_questions(self, decomposition_response: str) -> List[str]:
        """Extract sub-questions from the decomposition response."""
        sub_questions = []
        
        for line in decomposition_response.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                # Remove the number prefix and extract the question
                sub_question = re.sub(r'^\d+\.\s*', '', line).strip()
                if sub_question and len(sub_question) > 5:  # Basic quality check
                    sub_questions.append(sub_question)
        
        # Limit to max_sub_questions
        return sub_questions[:self.max_sub_questions]
    
    def _solve_sub_questions(self, sub_questions: List[str], original_question: str) -> List[str]:
        """Solve each sub-question iteratively, building context from previous answers."""
        sub_answers = []
        context = original_question + "\n\n"
        
        for i, sub_question in enumerate(sub_questions):
            if i == 0:
                # First sub-question: only use original context
                solve_prompt = f"""{context}Sub-question {i+1}: {sub_question}

Please answer this sub-question based on the information provided in the main question above.

Answer:"""
            else:
                # Subsequent sub-questions: include previous Q&A pairs
                previous_qa = "\n".join([
                    f"{j+1}. {sq} -> {sa}" 
                    for j, (sq, sa) in enumerate(zip(sub_questions[:i], sub_answers))
                ])
                
                solve_prompt = f"""{context}Previous sub-questions and answers:
{previous_qa}

Sub-question {i+1}: {sub_question}

Please answer this sub-question, taking into account the previous answers and the information from the main question.

Answer:"""
            
            sub_answer = api_agent(
                self.llm_model, 
                solve_prompt, 
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if sub_answer is None:
                print(f"Failed to get answer for sub-question {i+1}: {sub_question}")
                return []  # Return empty list if any sub-question fails
            
            sub_answers.append(sub_answer.strip())
        
        return sub_answers
    
    def _combine_answers(self, question: str, sub_questions: List[str], sub_answers: List[str]) -> Optional[str]:
        """Combine sub-answers to generate the final answer."""
        qa_pairs = "\n".join([
            f"{i+1}. {sq} -> {sa}" 
            for i, (sq, sa) in enumerate(zip(sub_questions, sub_answers))
        ])
        
        final_prompt = f"""{question}

Sub-questions and their answers:
{qa_pairs}

Based on the answers to these sub-questions, please provide the final answer to the main question. Use the information from the sub-answers to construct a comprehensive response. Enclose your ultimate answer in curly brackets {{}}.

Final Answer:"""
        
        return api_agent(
            self.llm_model, 
            final_prompt, 
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _compile_full_response(self, decomposition: str, sub_questions: List[str], 
                             sub_answers: List[str], final_answer: str) -> str:
        """Compile the complete response showing the full reasoning process."""
        response_parts = [
            "=== LEAST-TO-MOST DECOMPOSITION ===",
            decomposition,
            "",
            "=== SUB-QUESTION SOLUTIONS ===",
        ]
        
        for i, (sq, sa) in enumerate(zip(sub_questions, sub_answers)):
            response_parts.extend([
                f"Sub-question {i+1}: {sq}",
                f"Answer: {sa}",
                ""
            ])
        
        response_parts.extend([
            "=== FINAL ANSWER ===",
            final_answer if final_answer else "Failed to generate final answer"
        ])
        
        return "\n".join(response_parts)


class LeastToMostHoTStrategy(BaseStrategy):
    """
    Least-to-Most with Hard-of-Thought (grounding) strategy.
    Combines LtM decomposition with fact tagging and grounding.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.max_sub_questions = config.args.get('max_sub_questions', 3)
    
    def generate_response(self, question: str, dataset: str, few_shot_prompt: str = "", tail: str = "") -> Optional[str]:
        """Generate response using LtM with HoT grounding."""
        try:
            # Step 1: Create reformatted question with tags and decompose
            reformat_response = self._reformat_and_decompose(question, dataset)
            if not reformat_response:
                return None
            
            # Step 2: Extract reformatted question and sub-questions
            reformatted_question, sub_questions = self._extract_reformat_and_subquestions(reformat_response)
            if not reformatted_question or not sub_questions:
                return None
            
            # Step 3: Solve sub-questions with grounding
            sub_answers = self._solve_sub_questions_with_grounding(
                sub_questions, reformatted_question
            )
            if not sub_answers:
                return None
            
            # Step 4: Combine with grounding
            final_answer = self._combine_with_grounding(
                reformatted_question, sub_questions, sub_answers
            )
            
            # Compile full response
            full_response = self._compile_full_response_hot(
                reformat_response, reformatted_question, sub_questions, sub_answers, final_answer
            )
            
            return full_response
            
        except Exception as e:
            print(f"Error in LeastToMostHoTStrategy: {str(e)}")
            return None
    
    def _reformat_and_decompose(self, question: str, dataset: str) -> Optional[str]:
        """First create reformatted question with tags, then decompose."""
        last_sentence = extract_last_sentence(question, dataset)
        
        reformat_prompt = f"""{question}

Please re-generate this question with proper tags (<fact1></fact1>, <fact2></fact2>, <fact3></fact3>, etc.) around key phrases that are most relevant to answering {last_sentence}. Then break it down into 2-3 simpler sub-questions.

Format:
Reformatted Question: [question with <fact1></fact1>, <fact2></fact2>, <fact3></fact3>, etc. tags]

Sub-questions:
1. [First sub-question]
2. [Second sub-question]
3. [Third sub-question (if needed)]"""

        return api_agent(
            self.llm_model, 
            reformat_prompt, 
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _extract_reformat_and_subquestions(self, response: str) -> Tuple[str, List[str]]:
        """Extract reformatted question and sub-questions from response."""
        lines = response.split('\n')
        reformatted_question = ""
        sub_questions = []
        in_sub_questions = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('Reformatted Question:'):
                reformatted_question = line.replace('Reformatted Question:', '').strip()
            elif line.startswith('Sub-questions:'):
                in_sub_questions = True
            elif in_sub_questions and re.match(r'^\d+\.', line):
                sub_question = re.sub(r'^\d+\.\s*', '', line).strip()
                if sub_question and len(sub_question) > 5:
                    sub_questions.append(sub_question)
        
        return reformatted_question, sub_questions[:self.max_sub_questions]
    
    def _solve_sub_questions_with_grounding(self, sub_questions: List[str], 
                                          reformatted_question: str) -> List[str]:
        """Solve each sub-question with references to tagged facts."""
        sub_answers = []
        
        for i, sub_question in enumerate(sub_questions):
            if i == 0:
                solve_prompt = f"""Reformatted Question: {reformatted_question}

Sub-question {i+1}: {sub_question}

Please answer this sub-question using references to the tagged facts (<fact1></fact1>, <fact2></fact2>, <fact3></fact3>, etc.) from the reformatted question above. Include the relevant tags in your answer.

Answer:"""
            else:
                previous_qa = "\n".join([
                    f"{j+1}. {sq} -> {sa}" 
                    for j, (sq, sa) in enumerate(zip(sub_questions[:i], sub_answers))
                ])
                
                solve_prompt = f"""Reformatted Question: {reformatted_question}

Previous sub-questions and answers:
{previous_qa}

Sub-question {i+1}: {sub_question}

Please answer this sub-question using references to the tagged facts (<fact1></fact1>, <fact2></fact2>, <fact3></fact3>, etc.) and taking into account the previous answers.

Answer:"""
            
            sub_answer = api_agent(
                self.llm_model, 
                solve_prompt, 
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if sub_answer is None:
                return []
            
            sub_answers.append(sub_answer.strip())
        
        return sub_answers
    
    def _combine_with_grounding(self, reformatted_question: str, sub_questions: List[str], 
                               sub_answers: List[str]) -> Optional[str]:
        """Combine answers with grounding to tagged facts."""
        qa_pairs = "\n".join([
            f"{i+1}. {sq} -> {sa}" 
            for i, (sq, sa) in enumerate(zip(sub_questions, sub_answers))
        ])
        
        final_prompt = f"""Reformatted Question: {reformatted_question}

Sub-questions and their answers:
{qa_pairs}

Based on the tagged facts and the answers to these sub-questions, please provide the final answer. Use the fact tags (<fact1>, <fact2>, etc.) to show your grounding to the original question. Enclose your ultimate answer in curly brackets {{}}.

Answer:"""
        
        return api_agent(
            self.llm_model, 
            final_prompt, 
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _compile_full_response_hot(self, reformat_response: str, reformatted_question: str,
                                  sub_questions: List[str], sub_answers: List[str], 
                                  final_answer: str) -> str:
        """Compile the complete response for LtM+HoT."""
        response_parts = [
            "=== LEAST-TO-MOST + HARD-OF-THOUGHT ===",
            reformat_response,
            "",
            "=== GROUNDED SUB-QUESTION SOLUTIONS ===",
        ]
        
        for i, (sq, sa) in enumerate(zip(sub_questions, sub_answers)):
            response_parts.extend([
                f"Sub-question {i+1}: {sq}",
                f"Grounded Answer: {sa}",
                ""
            ])
        
        response_parts.extend([
            "=== FINAL GROUNDED ANSWER ===",
            final_answer if final_answer else "Failed to generate final answer"
        ])
        
        return "\n".join(response_parts)
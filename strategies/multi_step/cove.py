"""
Chain-of-Verification (CoVE) prompting strategy implementation.
Generates initial answer, creates verification questions, then revises based on verification.
"""

import re
from typing import List, Optional, Tuple

from ..base_strategy import BaseStrategy
from agents.api_agents import api_agent
from prompts.prompt_utils import extract_last_sentence


class ChainOfVerificationStrategy(BaseStrategy):
    """
    Chain-of-Verification prompting strategy.
    
    Process:
    1. Generate initial answer to the question
    2. Create verification questions to check the answer
    3. Answer each verification question
    4. Revise the original answer based on verification results
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_verification_questions = getattr(config.args, 'num_verification_questions', 3)
        self.revision_required = getattr(config.args, 'revision_required', True)
    
    def generate_response(self, question: str, dataset: str) -> Optional[str]:
        """Generate response using Chain-of-Verification prompting."""
        try:
            # Step 1: Generate initial answer
            initial_answer = self._generate_initial_answer(question, dataset)
            if not initial_answer:
                return None
            
            # Step 2: Create verification questions
            verification_questions_response = self._create_verification_questions(
                question, initial_answer
            )
            if not verification_questions_response:
                return None
            
            # Step 3: Extract verification questions
            verification_questions = self._extract_verification_questions(
                verification_questions_response
            )
            if not verification_questions:
                return None
            
            # Step 4: Answer verification questions
            verification_answers = self._answer_verification_questions(
                question, initial_answer, verification_questions
            )
            if not verification_answers:
                return None
            
            # Step 5: Revise answer based on verification
            final_answer = self._revise_answer_based_on_verification(
                question, initial_answer, verification_questions, verification_answers
            )
            
            # Compile full response
            full_response = self._compile_full_response(
                initial_answer, verification_questions_response, verification_questions,
                verification_answers, final_answer
            )
            
            return full_response
            
        except Exception as e:
            print(f"Error in ChainOfVerificationStrategy: {str(e)}")
            return None
    
    def _generate_initial_answer(self, question: str, dataset: str) -> Optional[str]:
        """Generate the initial answer to the question."""
        last_sentence = extract_last_sentence(question, dataset)
        
        initial_prompt = f"""{question}

Please provide your initial answer to this question: {last_sentence}

Provide detailed reasoning and your conclusion.

Initial Answer:"""

        return api_agent(
            self.llm_model,
            initial_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _create_verification_questions(self, question: str, initial_answer: str) -> Optional[str]:
        """Create verification questions to check the initial answer."""
        verification_prompt = f"""{question}

My initial answer was: {initial_answer}

To verify this answer, please generate {self.num_verification_questions} specific verification questions that would help check if my initial answer is correct or if there are any errors in my reasoning. These questions should:
1. Test key assumptions in the reasoning
2. Check for potential errors or oversights
3. Verify factual claims made in the answer

Verification Questions:
1. [First verification question]
2. [Second verification question]
3. [Third verification question (if needed)]"""

        return api_agent(
            self.llm_model,
            verification_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _extract_verification_questions(self, response: str) -> List[str]:
        """Extract verification questions from the response."""
        verification_questions = []
        
        for line in response.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                question_text = re.sub(r'^\d+\.\s*', '', line).strip()
                if question_text and len(question_text) > 10:
                    verification_questions.append(question_text)
        
        return verification_questions[:self.num_verification_questions]
    
    def _answer_verification_questions(self, question: str, initial_answer: str,
                                     verification_questions: List[str]) -> List[str]:
        """Answer each verification question."""
        verification_answers = []
        
        for i, verif_question in enumerate(verification_questions):
            verif_prompt = f"""{question}

Original initial answer: {initial_answer}

Verification question {i+1}: {verif_question}

Please answer this verification question carefully to check the accuracy of the initial answer. Be objective and thorough in your analysis.

Answer to verification question {i+1}:"""
            
            verif_answer = api_agent(
                self.llm_model,
                verif_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if verif_answer is not None:
                verification_answers.append(verif_answer.strip())
            else:
                print(f"Failed to get answer for verification question {i+1}")
                verification_answers.append("[Failed to generate verification answer]")
        
        return verification_answers
    
    def _revise_answer_based_on_verification(self, question: str, initial_answer: str,
                                           verification_questions: List[str],
                                           verification_answers: List[str]) -> Optional[str]:
        """Revise the initial answer based on verification results."""
        qa_pairs = "\n".join([
            f"{i+1}. {q} -> {a}"
            for i, (q, a) in enumerate(zip(verification_questions, verification_answers))
        ])
        
        revision_prompt = f"""{question}

Initial Answer: {initial_answer}

Verification Questions and Answers:
{qa_pairs}

Based on the verification process above, please provide a revised final answer. Consider:
1. Were any errors identified in the initial answer?
2. Do the verification results support or contradict the initial reasoning?
3. What corrections or improvements should be made?

If the initial answer was correct, confirm it with supporting reasoning. If there were errors, provide the corrected answer. Enclose your ultimate answer in curly brackets {{}}.

Revised Final Answer:"""
        
        return api_agent(
            self.llm_model,
            revision_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _compile_full_response(self, initial_answer: str, verification_questions_response: str,
                             verification_questions: List[str], verification_answers: List[str],
                             final_answer: str) -> str:
        """Compile the complete Chain-of-Verification response."""
        response_parts = [
            "=== CHAIN-OF-VERIFICATION ===",
            "",
            "=== INITIAL ANSWER ===",
            initial_answer,
            "",
            "=== VERIFICATION QUESTIONS GENERATION ===",
            verification_questions_response,
            "",
            "=== VERIFICATION PROCESS ===",
        ]
        
        for i, (q, a) in enumerate(zip(verification_questions, verification_answers)):
            response_parts.extend([
                f"Verification {i+1}: {q}",
                f"Answer: {a}",
                ""
            ])
        
        response_parts.extend([
            "=== REVISED FINAL ANSWER ===",
            final_answer if final_answer else "Failed to generate revised answer"
        ])
        
        return "\n".join(response_parts)


class ChainOfVerificationHoTStrategy(BaseStrategy):
    """
    Chain-of-Verification with Hard-of-Thought (grounding) strategy.
    Combines CoVE verification process with fact tagging and grounding.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_verification_questions = getattr(config.args, 'num_verification_questions', 3)
    
    def generate_response(self, question: str, dataset: str) -> Optional[str]:
        """Generate response using CoVE with HoT grounding."""
        try:
            # Step 1: Generate reformatted question with fact tags
            reformatted_response = self._reformat_question_with_tags(question, dataset)
            if not reformatted_response:
                return None
            
            # Step 2: Extract reformatted question
            reformatted_question = self._extract_reformatted_question(reformatted_response)
            if not reformatted_question:
                return None
            
            # Step 3: Generate initial answer with grounding
            initial_answer = self._generate_initial_answer_with_grounding(reformatted_question)
            if not initial_answer:
                return None
            
            # Step 4: Create verification questions with grounding awareness
            verification_questions_response = self._create_verification_questions_with_grounding(
                reformatted_question, initial_answer
            )
            if not verification_questions_response:
                return None
            
            # Step 5: Extract verification questions
            verification_questions = self._extract_verification_questions(verification_questions_response)
            if not verification_questions:
                return None
            
            # Step 6: Answer verification questions with grounding
            verification_answers = self._answer_verification_questions_with_grounding(
                reformatted_question, initial_answer, verification_questions
            )
            if not verification_answers:
                return None
            
            # Step 7: Revise with grounding
            final_answer = self._revise_answer_with_grounding(
                reformatted_question, initial_answer, verification_questions, verification_answers
            )
            
            # Compile full response
            full_response = self._compile_full_response_hot(
                reformatted_response, reformatted_question, initial_answer,
                verification_questions_response, verification_questions,
                verification_answers, final_answer
            )
            
            return full_response
            
        except Exception as e:
            print(f"Error in ChainOfVerificationHoTStrategy: {str(e)}")
            return None
    
    def _reformat_question_with_tags(self, question: str, dataset: str) -> Optional[str]:
        """Generate reformatted question with fact tags."""
        last_sentence = extract_last_sentence(question, dataset)
        
        reformat_prompt = f"""{question}

Please re-generate this question with proper tags (<fact1></fact1>, <fact2></fact2>, <fact3></fact3>, etc.) around key phrases that are most relevant to answering {last_sentence}.

Format:
Reformatted Question: [question with <fact1></fact1>, <fact2></fact2>, <fact3></fact3>, etc. tags]"""

        return api_agent(
            self.llm_model,
            reformat_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _extract_reformatted_question(self, response: str) -> str:
        """Extract the reformatted question from response."""
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('Reformatted Question:'):
                return line.replace('Reformatted Question:', '').strip()
        
        # If no explicit reformatted question found, return the whole response
        return response.strip()
    
    def _generate_initial_answer_with_grounding(self, reformatted_question: str) -> Optional[str]:
        """Generate initial answer with references to tagged facts."""
        initial_prompt = f"""Reformatted Question: {reformatted_question}

Please provide your initial answer to this question using references to the tagged facts (<fact1></fact1>, <fact2></fact2>, <fact3></fact3>, etc.) from the reformatted question above. Include the relevant tags in your reasoning and conclusion.

Initial Answer:"""

        return api_agent(
            self.llm_model,
            initial_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _create_verification_questions_with_grounding(self, reformatted_question: str,
                                                    initial_answer: str) -> Optional[str]:
        """Create verification questions that reference tagged facts."""
        verification_prompt = f"""Reformatted Question: {reformatted_question}

My initial answer was: {initial_answer}

To verify this answer, please generate {self.num_verification_questions} specific verification questions that would help check if my initial answer is correct or if there are any errors in my reasoning. These questions should reference the tagged facts when relevant and should:
1. Test key assumptions about the tagged facts
2. Check relationships between different facts
3. Verify the grounding of conclusions to the original facts

Verification Questions:
1. [First verification question]
2. [Second verification question]
3. [Third verification question (if needed)]"""

        return api_agent(
            self.llm_model,
            verification_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _answer_verification_questions_with_grounding(self, reformatted_question: str,
                                                    initial_answer: str,
                                                    verification_questions: List[str]) -> List[str]:
        """Answer verification questions with grounding to tagged facts."""
        verification_answers = []
        
        for i, verif_question in enumerate(verification_questions):
            verif_prompt = f"""Reformatted Question: {reformatted_question}

Original initial answer: {initial_answer}

Verification question {i+1}: {verif_question}

Please answer this verification question to check the accuracy of the initial answer. Use references to the tagged facts (<fact1></fact1>, <fact2></fact2>, <fact3></fact3>, etc.) when relevant. Be objective and thorough.

Answer to verification question {i+1}:"""
            
            verif_answer = api_agent(
                self.llm_model,
                verif_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if verif_answer is not None:
                verification_answers.append(verif_answer.strip())
            else:
                verification_answers.append("[Failed to generate grounded verification answer]")
        
        return verification_answers
    
    def _revise_answer_with_grounding(self, reformatted_question: str, initial_answer: str,
                                    verification_questions: List[str],
                                    verification_answers: List[str]) -> Optional[str]:
        """Revise answer based on verification with grounding."""
        qa_pairs = "\n".join([
            f"{i+1}. {q} -> {a}"
            for i, (q, a) in enumerate(zip(verification_questions, verification_answers))
        ])
        
        revision_prompt = f"""Reformatted Question: {reformatted_question}

Initial Answer: {initial_answer}

Verification Questions and Answers:
{qa_pairs}

Based on the verification process and the tagged facts (<fact1>, <fact2>, etc.), please provide a revised final answer. If the initial answer was correct, confirm it with proper grounding. If there were errors, provide the corrected answer with references to the relevant tagged facts. Enclose your ultimate answer in curly brackets {{}}.

Revised Final Answer:"""
        
        return api_agent(
            self.llm_model,
            revision_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _compile_full_response_hot(self, reformatted_response: str, reformatted_question: str,
                                  initial_answer: str, verification_questions_response: str,
                                  verification_questions: List[str], verification_answers: List[str],
                                  final_answer: str) -> str:
        """Compile the complete CoVE+HoT response."""
        response_parts = [
            "=== CHAIN-OF-VERIFICATION + HARD-OF-THOUGHT ===",
            "",
            "=== QUESTION REFORMATTING WITH FACT TAGS ===",
            reformatted_response,
            "",
            "=== INITIAL GROUNDED ANSWER ===",
            initial_answer,
            "",
            "=== GROUNDED VERIFICATION QUESTIONS ===",
            verification_questions_response,
            "",
            "=== GROUNDED VERIFICATION PROCESS ===",
        ]
        
        for i, (q, a) in enumerate(zip(verification_questions, verification_answers)):
            response_parts.extend([
                f"Verification {i+1}: {q}",
                f"Grounded Answer: {a}",
                ""
            ])
        
        response_parts.extend([
            "=== REVISED GROUNDED FINAL ANSWER ===",
            final_answer if final_answer else "Failed to generate revised grounded answer"
        ])
        
        return "\n".join(response_parts)
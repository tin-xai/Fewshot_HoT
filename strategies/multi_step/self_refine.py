"""
Self-Refine prompting strategy implementation.
Generates initial answer, self-critiques, then iteratively refines the answer.
"""

from typing import Optional

from ..base_strategy import BaseStrategy
from agents.api_agents import api_agent
from prompts.prompt_utils import extract_last_sentence


class SelfRefineStrategy(BaseStrategy):
    """
    Self-Refine prompting strategy.
    
    Process:
    1. Generate initial answer
    2. Self-critique the answer to identify weaknesses
    3. Refine the answer based on critique
    4. Optional second round of critique and refinement
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.max_refinement_rounds = getattr(config.args, 'max_refinement_rounds', 2)
        self.critique_required = getattr(config.args, 'critique_required', True)
    
    def generate_response(self, question: str, dataset: str) -> Optional[str]:
        """Generate response using Self-Refine prompting."""
        try:
            # Step 1: Generate initial answer
            initial_answer = self._generate_initial_answer(question, dataset)
            if not initial_answer:
                return None
            
            current_answer = initial_answer
            refinement_history = [("Initial", initial_answer)]
            
            # Step 2-4: Iterative critique and refinement
            for round_num in range(self.max_refinement_rounds):
                # Self-critique current answer
                critique = self._self_critique_answer(question, current_answer, round_num + 1)
                if not critique:
                    break
                
                # Refine based on critique
                refined_answer = self._refine_answer(question, current_answer, critique, round_num + 1)
                if not refined_answer:
                    break
                
                refinement_history.append((f"Critique {round_num + 1}", critique))
                refinement_history.append((f"Refinement {round_num + 1}", refined_answer))
                current_answer = refined_answer
            
            # Compile full response
            full_response = self._compile_full_response(refinement_history)
            
            return full_response
            
        except Exception as e:
            print(f"Error in SelfRefineStrategy: {str(e)}")
            return None
    
    def _generate_initial_answer(self, question: str, dataset: str) -> Optional[str]:
        """Generate the initial answer to the question."""
        last_sentence = extract_last_sentence(question, dataset)
        
        initial_prompt = f"""{question}

Please provide your initial answer to this question: {last_sentence}

Provide detailed reasoning and your conclusion. Be thorough but also prepare for the possibility that this answer might need refinement.

Initial Answer:"""

        return api_agent(
            self.llm_model,
            initial_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _self_critique_answer(self, question: str, current_answer: str, round_num: int) -> Optional[str]:
        """Perform self-critique on the current answer."""
        critique_prompt = f"""{question}

My current answer: {current_answer}

Please critically evaluate this answer (Round {round_num} critique). Consider:

1. **Logical consistency**: Are there any logical flaws or contradictions?
2. **Completeness**: Are there missing steps or considerations?
3. **Accuracy**: Are there any factual errors or misinterpretations?
4. **Clarity**: Is the reasoning clear and well-explained?
5. **Alternative perspectives**: Are there other ways to approach this problem?

Be specific about any issues you identify. If the answer is largely correct, note what works well but also identify any areas for potential improvement.

Self-Critique (Round {round_num}):"""

        return api_agent(
            self.llm_model,
            critique_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _refine_answer(self, question: str, current_answer: str, critique: str, round_num: int) -> Optional[str]:
        """Refine the answer based on the self-critique."""
        refine_prompt = f"""{question}

Current answer: {current_answer}

Self-critique: {critique}

Based on the critique above, please provide an improved and refined answer (Round {round_num} refinement). Address the identified issues and incorporate any suggested improvements. If the critique indicates the answer is already good, you may confirm it with minor improvements or additional clarity.

Refined Answer (Round {round_num}):"""

        return api_agent(
            self.llm_model,
            refine_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _compile_full_response(self, refinement_history: list) -> str:
        """Compile the complete self-refine response."""
        response_parts = [
            "=== SELF-REFINE PROCESS ===",
            ""
        ]
        
        for stage, content in refinement_history:
            response_parts.extend([
                f"=== {stage.upper()} ===",
                content,
                ""
            ])
        
        # Add final answer extraction
        if refinement_history:
            final_content = refinement_history[-1][1]
            response_parts.extend([
                "=== FINAL REFINED ANSWER ===",
                f"After {len([h for h in refinement_history if 'Refinement' in h[0]])} round(s) of refinement:",
                final_content
            ])
        
        return "\n".join(response_parts)


class SelfRefineIterativeStrategy(BaseStrategy):
    """
    Extended Self-Refine strategy with more sophisticated iterative improvement.
    Includes aspect-specific critique and targeted refinements.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.max_refinement_rounds = getattr(config.args, 'max_refinement_rounds', 3)
        self.critique_aspects = [
            'accuracy', 'completeness', 'clarity', 'logic', 'methodology'
        ]
    
    def generate_response(self, question: str, dataset: str) -> Optional[str]:
        """Generate response using iterative Self-Refine with aspect-specific critique."""
        try:
            # Step 1: Generate initial answer
            initial_answer = self._generate_initial_answer(question, dataset)
            if not initial_answer:
                return None
            
            current_answer = initial_answer
            detailed_history = {
                'initial': initial_answer,
                'rounds': []
            }
            
            # Iterative refinement with aspect-specific critique
            for round_num in range(self.max_refinement_rounds):
                round_data = {'round': round_num + 1, 'critiques': {}, 'refinements': {}}
                
                # Perform aspect-specific critiques
                aspect_critiques = self._perform_aspect_specific_critique(
                    question, current_answer, round_num + 1
                )
                round_data['critiques'] = aspect_critiques
                
                # Check if refinement is needed
                needs_refinement = self._assess_refinement_need(aspect_critiques)
                if not needs_refinement:
                    round_data['decision'] = 'No further refinement needed'
                    detailed_history['rounds'].append(round_data)
                    break
                
                # Perform targeted refinement
                refined_answer = self._perform_targeted_refinement(
                    question, current_answer, aspect_critiques, round_num + 1
                )
                
                if refined_answer:
                    round_data['refinements']['targeted'] = refined_answer
                    current_answer = refined_answer
                else:
                    round_data['decision'] = 'Refinement failed'
                    break
                
                detailed_history['rounds'].append(round_data)
            
            # Compile comprehensive response
            full_response = self._compile_iterative_response(detailed_history, current_answer)
            
            return full_response
            
        except Exception as e:
            print(f"Error in SelfRefineIterativeStrategy: {str(e)}")
            return None
    
    def _perform_aspect_specific_critique(self, question: str, current_answer: str, 
                                        round_num: int) -> dict:
        """Perform critique on specific aspects of the answer."""
        aspect_critiques = {}
        
        for aspect in self.critique_aspects:
            aspect_descriptions = {
                'accuracy': 'factual correctness and freedom from errors',
                'completeness': 'thoroughness and coverage of all necessary points',
                'clarity': 'clear explanation and understandable reasoning',
                'logic': 'logical consistency and valid reasoning steps',
                'methodology': 'appropriateness of approach and problem-solving method'
            }
            
            critique_prompt = f"""{question}

Current answer: {current_answer}

Please evaluate this answer specifically for {aspect} - {aspect_descriptions[aspect]}.

Focus only on this aspect and provide:
1. Current status (Good/Needs Improvement/Poor)
2. Specific observations about this aspect
3. Concrete suggestions for improvement if needed

{aspect.title()} Critique (Round {round_num}):"""
            
            critique = api_agent(
                self.llm_model,
                critique_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if critique:
                aspect_critiques[aspect] = critique.strip()
            
        return aspect_critiques
    
    def _assess_refinement_need(self, aspect_critiques: dict) -> bool:
        """Assess whether refinement is needed based on critiques."""
        if not aspect_critiques:
            return False
        
        # Simple heuristic: if any critique mentions "needs improvement" or "poor"
        # or contains improvement suggestions, refinement is needed
        refinement_keywords = [
            'needs improvement', 'poor', 'incorrect', 'error', 'missing',
            'unclear', 'confusing', 'incomplete', 'should', 'could improve'
        ]
        
        for critique in aspect_critiques.values():
            critique_lower = critique.lower()
            if any(keyword in critique_lower for keyword in refinement_keywords):
                return True
        
        return False
    
    def _perform_targeted_refinement(self, question: str, current_answer: str,
                                   aspect_critiques: dict, round_num: int) -> Optional[str]:
        """Perform refinement targeting specific aspects that need improvement."""
        # Combine all critiques with focus on improvement areas
        critique_summary = []
        for aspect, critique in aspect_critiques.items():
            critique_summary.append(f"{aspect.title()}: {critique}")
        
        combined_critiques = "\n\n".join(critique_summary)
        
        refinement_prompt = f"""{question}

Current answer: {current_answer}

Aspect-specific critiques:
{combined_critiques}

Based on these detailed critiques, please provide a refined answer that addresses the specific issues identified in each aspect. Focus particularly on:
- Fixing any accuracy issues
- Adding missing information for completeness
- Improving clarity where needed
- Strengthening logical connections
- Refining methodology if appropriate

Enclose your ultimate answer in curly brackets {{}} if this is the final version.

Targeted Refined Answer (Round {round_num}):"""
        
        return api_agent(
            self.llm_model,
            refinement_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _compile_iterative_response(self, detailed_history: dict, final_answer: str) -> str:
        """Compile the comprehensive iterative self-refine response."""
        response_parts = [
            "=== ITERATIVE SELF-REFINE PROCESS ===",
            "",
            "=== INITIAL ANSWER ===",
            detailed_history['initial'],
            ""
        ]
        
        for round_data in detailed_history['rounds']:
            response_parts.extend([
                f"=== ROUND {round_data['round']} ===",
                ""
            ])
            
            # Add aspect-specific critiques
            if round_data['critiques']:
                response_parts.append("--- Aspect-Specific Critiques ---")
                for aspect, critique in round_data['critiques'].items():
                    response_parts.extend([
                        f"{aspect.title()}: {critique}",
                        ""
                    ])
            
            # Add refinements or decisions
            if 'decision' in round_data:
                response_parts.extend([
                    f"Decision: {round_data['decision']}",
                    ""
                ])
            
            if round_data.get('refinements'):
                response_parts.extend([
                    "--- Targeted Refinement ---",
                    round_data['refinements']['targeted'],
                    ""
                ])
        
        response_parts.extend([
            "=== FINAL REFINED ANSWER ===",
            f"After {len(detailed_history['rounds'])} round(s) of iterative refinement:",
            final_answer
        ])
        
        return "\n".join(response_parts)
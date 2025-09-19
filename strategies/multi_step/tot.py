"""
Tree-of-Thought (ToT) prompting strategy implementation.
Explores multiple reasoning paths and selects the best one.
"""

import re
from typing import List, Optional

from ..base_strategy import BaseStrategy
from agents.api_agents import api_agent
from prompts.prompt_utils import extract_last_sentence


class TreeOfThoughtStrategy(BaseStrategy):
    """
    Tree-of-Thought prompting strategy.
    
    Process:
    1. Generate multiple different reasoning paths/approaches
    2. Execute each reasoning path to get solutions
    3. Evaluate all solutions and select the best one
    4. Provide final answer based on the best reasoning path
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_paths = getattr(config.args, 'num_paths', 3)
        self.evaluation_required = getattr(config.args, 'evaluation_required', True)
    
    def generate_response(self, question: str, dataset: str) -> Optional[str]:
        """Generate response using Tree-of-Thought prompting."""
        try:
            # Step 1: Generate multiple reasoning paths
            paths_response = self._generate_reasoning_paths(question, dataset)
            if not paths_response:
                return None
            
            # Step 2: Extract reasoning paths
            paths = self._extract_paths(paths_response)
            if not paths:
                return None
            
            # Step 3: Execute each reasoning path
            path_solutions = self._execute_reasoning_paths(question, paths)
            if not path_solutions:
                return None
            
            # Step 4: Evaluate and select best solution
            final_answer = self._evaluate_and_select_best(question, paths, path_solutions)
            
            # Compile full response
            full_response = self._compile_full_response(
                paths_response, paths, path_solutions, final_answer
            )
            
            return full_response
            
        except Exception as e:
            print(f"Error in TreeOfThoughtStrategy: {str(e)}")
            return None
    
    def _generate_reasoning_paths(self, question: str, dataset: str) -> Optional[str]:
        """Generate multiple reasoning approaches/paths."""
        last_sentence = extract_last_sentence(question, dataset)
        
        paths_prompt = f"""{question}

I need to explore different reasoning paths to solve this problem: {last_sentence}

Please generate {self.num_paths} different reasoning approaches/paths to solve this question. For each path, provide a brief but distinct reasoning strategy.

Format:
Path 1: [Reasoning approach 1]
Path 2: [Reasoning approach 2]
Path 3: [Reasoning approach 3]"""
        
        return api_agent(
            self.llm_model, 
            paths_prompt, 
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _extract_paths(self, paths_response: str) -> List[str]:
        """Extract reasoning paths from the response."""
        paths = []
        
        for line in paths_response.split('\n'):
            line = line.strip()
            if re.match(r'^Path \d+:', line):
                path = re.sub(r'^Path \d+:\s*', '', line).strip()
                if path and len(path) > 10:  # Basic quality check
                    paths.append(path)
        
        return paths[:self.num_paths]
    
    def _execute_reasoning_paths(self, question: str, paths: List[str]) -> List[str]:
        """Execute each reasoning path to get solutions."""
        path_solutions = []
        
        for i, path in enumerate(paths):
            solve_prompt = f"""{question}

Reasoning Path {i+1}: {path}

Following this specific reasoning approach, please solve the question step by step. Provide your detailed reasoning and conclusion.

Solution for Path {i+1}:"""
            
            solution = api_agent(
                self.llm_model, 
                solve_prompt, 
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if solution is not None:
                path_solutions.append(solution.strip())
            else:
                print(f"Failed to get solution for Path {i+1}")
                # Add placeholder to maintain indexing
                path_solutions.append(f"[Failed to generate solution for Path {i+1}]")
        
        return path_solutions
    
    def _evaluate_and_select_best(self, question: str, paths: List[str], 
                                 path_solutions: List[str]) -> Optional[str]:
        """Evaluate all reasoning paths and select the best one."""
        # Create evaluation text with all paths and solutions
        evaluation_text = []
        for i, (path, solution) in enumerate(zip(paths, path_solutions)):
            evaluation_text.append(f"Path {i+1}: {path}")
            evaluation_text.append(f"Solution {i+1}: {solution}")
            evaluation_text.append("")
        
        evaluation_prompt = f"""{question}

I have explored multiple reasoning paths:

{chr(10).join(evaluation_text)}

Please evaluate these different reasoning paths and their solutions. Consider:
1. Logical consistency and correctness
2. Completeness of reasoning
3. Clarity of explanation
4. Accuracy of the final conclusion

Select the most logical and accurate reasoning path, then provide the final answer based on the best path. Enclose your ultimate answer in curly brackets {{}}.

Evaluation and Final Answer:"""
        
        return api_agent(
            self.llm_model, 
            evaluation_prompt, 
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _compile_full_response(self, paths_response: str, paths: List[str], 
                             path_solutions: List[str], final_answer: str) -> str:
        """Compile the complete Tree-of-Thought response."""
        response_parts = [
            "=== TREE-OF-THOUGHT REASONING ===",
            "Generated Paths:",
            paths_response,
            "",
            "=== PATH EXECUTIONS ===",
        ]
        
        for i, (path, solution) in enumerate(zip(paths, path_solutions)):
            response_parts.extend([
                f"Path {i+1}: {path}",
                f"Solution {i+1}: {solution}",
                ""
            ])
        
        response_parts.extend([
            "=== EVALUATION AND FINAL ANSWER ===",
            final_answer if final_answer else "Failed to generate final evaluation"
        ])
        
        return "\n".join(response_parts)


class TreeOfThoughtIterativeStrategy(BaseStrategy):
    """
    Extended Tree-of-Thought with iterative refinement.
    Builds reasoning trees with multiple levels of branching.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_paths = getattr(config.args, 'num_paths', 3)
        self.max_depth = getattr(config.args, 'max_depth', 2)
        self.branching_factor = getattr(config.args, 'branching_factor', 2)
    
    def generate_response(self, question: str, dataset: str) -> Optional[str]:
        """Generate response using iterative Tree-of-Thought."""
        try:
            # Start with initial reasoning paths
            paths = self._generate_initial_paths(question, dataset)
            if not paths:
                return None
            
            # Build reasoning tree through iterations
            reasoning_tree = self._build_reasoning_tree(question, paths)
            
            # Select best path through the tree
            final_answer = self._select_best_path_from_tree(question, reasoning_tree)
            
            # Compile response
            full_response = self._compile_tree_response(reasoning_tree, final_answer)
            
            return full_response
            
        except Exception as e:
            print(f"Error in TreeOfThoughtIterativeStrategy: {str(e)}")
            return None
    
    def _generate_initial_paths(self, question: str, dataset: str) -> List[str]:
        """Generate initial set of reasoning paths."""
        last_sentence = extract_last_sentence(question, dataset)
        
        paths_prompt = f"""{question}

To solve this problem: {last_sentence}, I need to explore different high-level approaches.

Generate {self.num_paths} distinct reasoning strategies, focusing on different angles or methods:

Format:
1. [Strategy 1]
2. [Strategy 2]
3. [Strategy 3]"""
        
        response = api_agent(
            self.llm_model, 
            paths_prompt, 
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        if not response:
            return []
        
        paths = []
        for line in response.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                path = re.sub(r'^\d+\.\s*', '', line).strip()
                if path:
                    paths.append(path)
        
        return paths[:self.num_paths]
    
    def _build_reasoning_tree(self, question: str, initial_paths: List[str]) -> dict:
        """Build a reasoning tree with multiple levels."""
        tree = {'question': question, 'levels': []}
        
        current_paths = initial_paths
        
        for depth in range(self.max_depth):
            level_results = []
            
            for i, path in enumerate(current_paths):
                # Execute current path
                execution = self._execute_path_step(question, path, depth)
                
                # Generate refined paths if not at max depth
                refined_paths = []
                if depth < self.max_depth - 1:
                    refined_paths = self._refine_path(question, path, execution)
                
                level_results.append({
                    'path': path,
                    'execution': execution,
                    'refined_paths': refined_paths
                })
            
            tree['levels'].append(level_results)
            
            # Prepare paths for next iteration
            next_paths = []
            for result in level_results:
                next_paths.extend(result['refined_paths'][:self.branching_factor])
            
            current_paths = next_paths
            if not current_paths:
                break
        
        return tree
    
    def _execute_path_step(self, question: str, path: str, depth: int) -> Optional[str]:
        """Execute a reasoning path at a specific depth."""
        execute_prompt = f"""{question}

Reasoning approach: {path}

Execute this reasoning approach step by step (Depth {depth+1}). Provide detailed reasoning and intermediate conclusions.

Execution:"""
        
        return api_agent(
            self.llm_model, 
            execute_prompt, 
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _refine_path(self, question: str, original_path: str, execution: str) -> List[str]:
        """Refine a reasoning path based on its execution."""
        refine_prompt = f"""{question}

Original approach: {original_path}
Execution result: {execution}

Based on this execution, suggest {self.branching_factor} refined or alternative approaches that could improve the reasoning:

1. [Refined approach 1]
2. [Refined approach 2]"""
        
        response = api_agent(
            self.llm_model, 
            refine_prompt, 
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        if not response:
            return []
        
        refined_paths = []
        for line in response.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                path = re.sub(r'^\d+\.\s*', '', line).strip()
                if path:
                    refined_paths.append(path)
        
        return refined_paths
    
    def _select_best_path_from_tree(self, question: str, tree: dict) -> Optional[str]:
        """Select the best reasoning path from the complete tree."""
        # Compile all reasoning paths and executions
        all_reasoning = []
        
        for level_idx, level in enumerate(tree['levels']):
            for path_idx, result in enumerate(level):
                all_reasoning.append(
                    f"Level {level_idx+1}, Path {path_idx+1}: {result['path']}\n"
                    f"Execution: {result['execution']}\n"
                )
        
        selection_prompt = f"""{question}

Complete reasoning tree exploration:

{chr(10).join(all_reasoning)}

Analyze all the reasoning paths and executions above. Select the most promising line of reasoning and provide the final answer based on the best approach. Enclose your ultimate answer in curly brackets {{}}.

Final Selection and Answer:"""
        
        return api_agent(
            self.llm_model, 
            selection_prompt, 
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _compile_tree_response(self, tree: dict, final_answer: str) -> str:
        """Compile the complete iterative Tree-of-Thought response."""
        response_parts = [
            "=== ITERATIVE TREE-OF-THOUGHT ===",
            f"Question: {tree['question']}",
            ""
        ]
        
        for level_idx, level in enumerate(tree['levels']):
            response_parts.append(f"--- LEVEL {level_idx+1} ---")
            
            for path_idx, result in enumerate(level):
                response_parts.extend([
                    f"Path {path_idx+1}: {result['path']}",
                    f"Execution: {result['execution']}",
                ])
                
                if result['refined_paths']:
                    response_parts.append(f"Refined paths: {', '.join(result['refined_paths'])}")
                
                response_parts.append("")
        
        response_parts.extend([
            "=== FINAL SELECTION ===",
            final_answer if final_answer else "Failed to select final answer"
        ])
        
        return "\n".join(response_parts)
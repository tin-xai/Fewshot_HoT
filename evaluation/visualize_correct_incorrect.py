#!/usr/bin/env python3
"""
Visualization Script for Evaluation Results
===========================================

Creates an HTML visualization showing side-by-side comparison:
- Left: Examples where File A is CORRECT
- Right: Same examples where File B is INCORRECT

Usage:
    python visualize_correct_incorrect.py \
        --file1 evaluation_results/model1_rule_eval.csv \
        --file2 evaluation_results/model2_rule_eval.csv \
        --output comparison.html
"""

import argparse
import pandas as pd
from typing import List, Dict, Tuple
import os


class ResultVisualizer:
    """Generate HTML visualizations for evaluation results."""
    
    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Results Comparison</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            color: #2c3e50;
            margin-bottom: 20px;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        
        .metric-card.correct {{
            border-left-color: #27ae60;
        }}
        
        .metric-card.incorrect {{
            border-left-color: #e74c3c;
        }}
        
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .metric-label {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 5px;
        }}
        
        .comparison-pair {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .column {{
            position: relative;
        }}
        
        .column-header {{
            font-size: 16px;
            font-weight: bold;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 15px;
            text-align: center;
        }}
        
        .correct-header {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        
        .incorrect-header {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        
        .vs-divider {{
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            background: #6c757d;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            z-index: 10;
        }}
        
        .example {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        
        .example-id {{
            font-size: 12px;
            color: #6c757d;
            margin-bottom: 10px;
        }}
        
        .question {{
            background: white;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 10px;
            border-left: 3px solid #3498db;
        }}
        
        .question-label {{
            font-size: 12px;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 5px;
        }}
        
        .answer-section {{
            display: grid;
            gap: 8px;
            margin-top: 10px;
        }}
        
        .answer {{
            padding: 10px;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        .extracted {{
            background: #e3f2fd;
            border-left: 3px solid #2196f3;
        }}
        
        .ground-truth {{
            background: #f3e5f5;
            border-left: 3px solid #9c27b0;
        }}
        
        .model-answer {{
            background: #fff3e0;
            border-left: 3px solid #ff9800;
            max-height: 200px;
            overflow-y: auto;
        }}
        
        .answer-label {{
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 5px;
            opacity: 0.7;
        }}
        
        .file-info {{
            background: #e3f2fd;
            padding: 10px 15px;
            border-radius: 6px;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        
        .comparison-mode {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }}
        
        .filter-controls {{
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }}
        
        .filter-btn {{
            padding: 8px 16px;
            border: 1px solid #dee2e6;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .filter-btn:hover {{
            background: #e9ecef;
        }}
        
        .filter-btn.active {{
            background: #3498db;
            color: white;
            border-color: #3498db;
        }}
        
        .section-title {{
            font-size: 24px;
            font-weight: bold;
            margin: 30px 0 20px 0;
            color: #2c3e50;
        }}
        
        .tag-unpaired {{
            background: #ffcccc;
            border: 1px solid #ff6666;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
        }}
        
        .tag-fact1 {{ background: #ffe4cc; border: 1px solid #ff9933; }}
        .tag-fact2 {{ background: #cce6ff; border: 1px solid #3399ff; }}
        .tag-fact3 {{ background: #e6ccff; border: 1px solid #9933ff; }}
        .tag-fact4 {{ background: #ccffcc; border: 1px solid #33cc33; }}
        .tag-fact5 {{ background: #ffccf2; border: 1px solid #ff33cc; }}
        .tag-fact6 {{ background: #ffffcc; border: 1px solid #cccc33; }}
        .tag-fact7 {{ background: #cce6ff; border: 1px solid #3366ff; }}
        .tag-fact8 {{ background: #ffcce6; border: 1px solid #ff3399; }}
        .tag-fact9 {{ background: #e6ffcc; border: 1px solid #99cc33; }}
        .tag-fact10 {{ background: #ffe6cc; border: 1px solid #ff9966; }}
        .tag-fact11 {{ background: #d4f1f9; border: 1px solid #5dade2; }}
        .tag-fact12 {{ background: #f9e79f; border: 1px solid #f4d03f; }}
        .tag-fact13 {{ background: #d7bde2; border: 1px solid #a569bd; }}
        .tag-fact14 {{ background: #a9dfbf; border: 1px solid #52be80; }}
        .tag-fact15 {{ background: #f5b7b1; border: 1px solid #ec7063; }}
        
        .tag-context {{ background: #d6eaf8; border: 1px solid #85c1e9; }}
        .tag-assumption {{ background: #fdebd0; border: 1px solid #f8c471; }}
        .tag-conclusion {{ background: #d5f4e6; border: 1px solid #82e0aa; }}
        .tag-evidence {{ background: #ebdef0; border: 1px solid #bb8fce; }}
        
        .tag-other {{ background: #e8e8e8; border: 1px solid #aaaaaa; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .stat-box {{
            background: #e7f3ff;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #0066cc;
        }}
        
        .stat-box h4 {{
            margin: 0 0 10px 0;
            color: #0066cc;
            font-size: 14px;
        }}
        
        .stat-box .value {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        @media (max-width: 768px) {{
            .comparison-pair {{
                grid-template-columns: 1fr;
            }}
            
            .metrics {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Evaluation Results Comparison</h1>
        
        <div class="comparison-mode">
            <div>
                <div class="file-info">📁 File 1: {file1_name}</div>
                <div class="file-info">📁 File 2: {file2_name}</div>
            </div>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{total1}</div>
                <div class="metric-label">Total Examples (File 1)</div>
            </div>
            <div class="metric-card correct">
                <div class="metric-value">{correct1}</div>
                <div class="metric-label">Correct (File 1)</div>
            </div>
            <div class="metric-card incorrect">
                <div class="metric-value">{incorrect1}</div>
                <div class="metric-label">Incorrect (File 1)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{accuracy1}%</div>
                <div class="metric-label">Accuracy (File 1)</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{total2}</div>
                <div class="metric-label">Total Examples (File 2)</div>
            </div>
            <div class="metric-card correct">
                <div class="metric-value">{correct2}</div>
                <div class="metric-label">Correct (File 2)</div>
            </div>
            <div class="metric-card incorrect">
                <div class="metric-value">{incorrect2}</div>
                <div class="metric-label">Incorrect (File 2)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{accuracy2}%</div>
                <div class="metric-label">Accuracy (File 2)</div>
            </div>
        </div>
    </div>
    
    <div style="padding: 0 20px;">
        <h2 class="section-title">📊 Analysis Statistics</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <h4>File A Avg Characters</h4>
                <div class="value">{avg_chars_file1}</div>
            </div>
            <div class="stat-box">
                <h4>File B Avg Characters</h4>
                <div class="value">{avg_chars_file2}</div>
            </div>
            <div class="stat-box">
                <h4>File A Avg Tags</h4>
                <div class="value">{avg_tags_file1}</div>
            </div>
            <div class="stat-box">
                <h4>File B Avg Tags</h4>
                <div class="value">{avg_tags_file2}</div>
            </div>
        </div>
        
        <h2 class="section-title">✅ File A Correct → ❌ File B Incorrect ({a_correct_b_incorrect_count} examples)</h2>
        {a_correct_b_incorrect}
        
        <h2 class="section-title">❌ File A Incorrect → ✅ File B Correct ({a_incorrect_b_correct_count} examples)</h2>
        {a_incorrect_b_correct}
        
        <h2 class="section-title">✅ Both Correct ({both_correct_count} examples)</h2>
        {both_correct}
        
        <h2 class="section-title">❌ Both Incorrect ({both_incorrect_count} examples)</h2>
        {both_incorrect}
    </div>
</body>
</html>"""
    
    COMPARISON_PAIR_TEMPLATE = """
    <div class="comparison-pair">
        <div class="column">
            <div class="column-header {left_status_class}">
                {left_status} File A: {file1_name}
            </div>
            <div class="example">
                <div class="example-id">ID: {id}</div>
                
                <div class="question">
                    <div class="question-label">Question</div>
                    {question}
                </div>
                
                <div class="answer-section">
                    <div class="extracted answer">
                        <div class="answer-label">Extracted Answer</div>
                        {extracted1}
                    </div>
                    
                    <div class="ground-truth answer">
                        <div class="answer-label">Ground Truth</div>
                        {ground_truth}
                    </div>
                    
                    <details>
                        <summary style="cursor: pointer; padding: 5px; background: #f8f9fa; border-radius: 4px;">
                            📝 Full Model Answer
                        </summary>
                        <div class="model-answer answer" style="margin-top: 10px;">
                            {model_answer1}
                        </div>
                    </details>
                </div>
            </div>
        </div>
        
        <div class="column">
            <div class="column-header {right_status_class}">
                {right_status} File B: {file2_name}
            </div>
            <div class="example">
                <div class="example-id">ID: {id}</div>
                
                <div class="question">
                    <div class="question-label">Question</div>
                    {question}
                </div>
                
                <div class="answer-section">
                    <div class="extracted answer">
                        <div class="answer-label">Extracted Answer</div>
                        {extracted2}
                    </div>
                    
                    <div class="ground-truth answer">
                        <div class="answer-label">Ground Truth</div>
                        {ground_truth}
                    </div>
                    
                    <details>
                        <summary style="cursor: pointer; padding: 5px; background: #f8f9fa; border-radius: 4px;">
                            📝 Full Model Answer
                        </summary>
                        <div class="model-answer answer" style="margin-top: 10px;">
                            {model_answer2}
                        </div>
                    </details>
                </div>
            </div>
        </div>
    </div>
    """
    
    def __init__(self):
        pass
    
    def load_results(self, filepath: str) -> pd.DataFrame:
        """Load evaluation results from CSV."""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} examples from {filepath}")
        return df
    
    def _count_tags(self, text: str) -> int:
        """Count the number of tags in text."""
        if not text:
            return 0
        # Count opening tags like <fact1>, <tag>, etc.
        import re
        tags = re.findall(r'<[^/>][^>]*>', text)
        return len(tags)
    
    def _get_tag_class(self, tag_name: str) -> str:
        """Get CSS class for a tag based on its name."""
        tag_lower = tag_name.lower()
        
        # Check for fact tags
        if tag_lower.startswith('fact'):
            # Extract number if present
            import re
            match = re.search(r'fact(\d+)', tag_lower)
            if match:
                num = int(match.group(1))
                if num <= 15:
                    return f'tag-fact{num}'
        
        # Check for semantic tags
        semantic_tags = ['context', 'assumption', 'conclusion', 'evidence']
        for semantic_tag in semantic_tags:
            if tag_lower == semantic_tag or tag_lower.startswith(semantic_tag):
                return f'tag-{semantic_tag}'
        
        # Default for other tags
        return 'tag-other'
    
    def _highlight_tags(self, text: str) -> str:
        """Highlight tags in the text with special formatting and different colors."""
        if not text:
            return ""
        
        import re
        
        # Process text in segments to avoid double-escaping
        result_parts = []
        last_end = 0
        
        # First, find all paired tags
        for match in re.finditer(r'<([a-zA-Z0-9_]+)>([^<]*?)</\1>', text):
            # Add text before the tag (escaped)
            if match.start() > last_end:
                before_text = text[last_end:match.start()]
                escaped_before = (before_text
                                 .replace('&', '&amp;')
                                 .replace('<', '&lt;')
                                 .replace('>', '&gt;')
                                 .replace('"', '&quot;')
                                 .replace("'", '&#39;'))
                result_parts.append(escaped_before)
            
            # Add highlighted tag content
            tag_name = match.group(1)
            content = match.group(2)
            tag_class = self._get_tag_class(tag_name)
            # Keep the original content, just wrap it with colored span
            result_parts.append(f'<span class="{tag_class}" style="padding: 2px 4px; border-radius: 3px;">{content}</span>')
            
            last_end = match.end()
        
        # Add remaining text after last tag
        if last_end < len(text):
            remaining = text[last_end:]
            # Find unpaired tags in remaining text
            remaining = re.sub(
                r'<([a-zA-Z0-9_]+)(?:[^>]*)>',
                lambda m: f'<span class="tag-unpaired" style="padding: 2px 4px; border-radius: 3px; font-family: monospace;">&lt;{m.group(1)}&gt;</span>',
                remaining
            )
            # Escape other special characters
            remaining = (remaining
                        .replace('&', '&amp;')
                        .replace('"', '&quot;')
                        .replace("'", '&#39;'))
            result_parts.append(remaining)
        
        text = ''.join(result_parts)
        
        # Replace newlines with <br>
        text = text.replace('\n', '<br>')
        
        return text
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters and highlight tags."""
        return self._highlight_tags(text)
    
    def generate_comparison_pair_html(self, row1: pd.Series, row2: pd.Series, 
                                     file1_name: str, file2_name: str,
                                     left_correct: bool, right_correct: bool, 
                                     example_id: int) -> str:
        """Generate HTML for a side-by-side comparison pair."""
        left_status = "✅" if left_correct else "❌"
        right_status = "✅" if right_correct else "❌"
        left_class = "correct-header" if left_correct else "incorrect-header"
        right_class = "correct-header" if right_correct else "incorrect-header"
        
        # Safely get values with fallbacks
        def safe_get(row, key, default='N/A'):
            try:
                return row[key] if key in row.index else default
            except:
                return default
        
        return self.COMPARISON_PAIR_TEMPLATE.format(
            id=example_id,
            file1_name=file1_name,
            file2_name=file2_name,
            left_status=left_status,
            right_status=right_status,
            left_status_class=left_class,
            right_status_class=right_class,
            question=self._escape_html(str(safe_get(row1, 'question'))),
            extracted1=self._escape_html(str(safe_get(row1, 'extracted_answer'))),
            extracted2=self._escape_html(str(safe_get(row2, 'extracted_answer'))),
            ground_truth=self._escape_html(str(safe_get(row1, 'ground_truth'))),
            model_answer1=self._escape_html(str(safe_get(row1, 'model_answer', safe_get(row1, 'answer')))),
            model_answer2=self._escape_html(str(safe_get(row2, 'model_answer', safe_get(row2, 'answer'))))
        )
    
    def generate_comparison_html(self, file1_path: str, file2_path: str) -> str:
        """Generate comparison HTML for two result files."""
        df1 = self.load_results(file1_path)
        df2 = self.load_results(file2_path)
        
        file1_name = os.path.basename(file1_path)
        file2_name = os.path.basename(file2_path)
        
        # Check if 'id' column exists, if not create it from index
        if 'id' not in df1.columns:
            df1['id'] = df1.index
        if 'id' not in df2.columns:
            df2['id'] = df2.index
        
        # Merge on ID to compare same examples
        df1_indexed = df1.set_index('id')
        df2_indexed = df2.set_index('id')
        
        # Find common IDs
        common_ids = df1_indexed.index.intersection(df2_indexed.index)
        
        print(f"Found {len(common_ids)} common examples between the two files")
        
        # Categorize comparisons
        a_correct_b_incorrect = []
        a_incorrect_b_correct = []
        both_correct = []
        both_incorrect = []
        
        for id in common_ids:
            row1 = df1_indexed.loc[id]
            row2 = df2_indexed.loc[id]
            
            is_correct_1 = row1['is_correct']
            is_correct_2 = row2['is_correct']
            
            html = self.generate_comparison_pair_html(
                row1, row2, file1_name, file2_name, 
                is_correct_1, is_correct_2, id
            )
            
            if is_correct_1 and not is_correct_2:
                a_correct_b_incorrect.append(html)
            elif not is_correct_1 and is_correct_2:
                a_incorrect_b_correct.append(html)
            elif is_correct_1 and is_correct_2:
                both_correct.append(html)
            else:
                both_incorrect.append(html)
        
        # Calculate metrics
        total1 = len(df1)
        correct1 = df1['is_correct'].sum()
        incorrect1 = total1 - correct1
        accuracy1 = (correct1 / total1 * 100) if total1 > 0 else 0
        
        total2 = len(df2)
        correct2 = df2['is_correct'].sum()
        incorrect2 = total2 - correct2
        accuracy2 = (correct2 / total2 * 100) if total2 > 0 else 0
        
        # Calculate character and tag statistics
        def safe_get_text(row, key):
            try:
                val = row.get(key, row.get('answer', ''))
                return str(val) if val and val != 'N/A' else ''
            except:
                return ''
        
        # Calculate for file 1
        char_counts_1 = [len(safe_get_text(row, 'model_answer')) for _, row in df1.iterrows()]
        tag_counts_1 = [self._count_tags(safe_get_text(row, 'model_answer')) for _, row in df1.iterrows()]
        
        avg_chars_1 = sum(char_counts_1) / len(char_counts_1) if char_counts_1 else 0
        avg_tags_1 = sum(tag_counts_1) / len(tag_counts_1) if tag_counts_1 else 0
        
        # Calculate for file 2
        char_counts_2 = [len(safe_get_text(row, 'model_answer')) for _, row in df2.iterrows()]
        tag_counts_2 = [self._count_tags(safe_get_text(row, 'model_answer')) for _, row in df2.iterrows()]
        
        avg_chars_2 = sum(char_counts_2) / len(char_counts_2) if char_counts_2 else 0
        avg_tags_2 = sum(tag_counts_2) / len(tag_counts_2) if tag_counts_2 else 0
        
        # Generate final HTML
        html = self.HTML_TEMPLATE.format(
            file1_name=file1_name,
            file2_name=file2_name,
            total1=total1,
            correct1=correct1,
            incorrect1=incorrect1,
            accuracy1=f"{accuracy1:.2f}",
            total2=total2,
            correct2=correct2,
            incorrect2=incorrect2,
            accuracy2=f"{accuracy2:.2f}",
            avg_chars_file1=f"{avg_chars_1:.0f}",
            avg_chars_file2=f"{avg_chars_2:.0f}",
            avg_tags_file1=f"{avg_tags_1:.2f}",
            avg_tags_file2=f"{avg_tags_2:.2f}",
            a_correct_b_incorrect_count=len(a_correct_b_incorrect),
            a_incorrect_b_correct_count=len(a_incorrect_b_correct),
            both_correct_count=len(both_correct),
            both_incorrect_count=len(both_incorrect),
            a_correct_b_incorrect='\n'.join(a_correct_b_incorrect) if a_correct_b_incorrect else '<p style="color: #6c757d; text-align: center; padding: 20px;">No examples where File A is correct and File B is incorrect</p>',
            a_incorrect_b_correct='\n'.join(a_incorrect_b_correct) if a_incorrect_b_correct else '<p style="color: #6c757d; text-align: center; padding: 20px;">No examples where File A is incorrect and File B is correct</p>',
            both_correct='\n'.join(both_correct) if both_correct else '<p style="color: #6c757d; text-align: center; padding: 20px;">No examples where both are correct</p>',
            both_incorrect='\n'.join(both_incorrect) if both_incorrect else '<p style="color: #6c757d; text-align: center; padding: 20px;">No examples where both are incorrect</p>'
        )
        
        return html
    


def main():
    parser = argparse.ArgumentParser(description="Visualize Side-by-Side Evaluation Results")
    parser.add_argument('--file1', type=str, required=True,
                       help='Path to first evaluation CSV file (File A)')
    parser.add_argument('--file2', type=str, required=True,
                       help='Path to second evaluation CSV file (File B)')
    parser.add_argument('--output', type=str, default='evaluation_comparison.html',
                       help='Output HTML file path')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ResultVisualizer()
    
    # Generate HTML
    print(f"\nGenerating side-by-side comparison visualization...")
    print(f"File A: {args.file1}")
    print(f"File B: {args.file2}")
    html = visualizer.generate_comparison_html(args.file1, args.file2)
    
    # Save HTML
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✓ Visualization saved to: {args.output}")
    print(f"📂 Open the file in your browser to view the results")
    print(f"\nThe visualization shows:")
    print(f"  - File A correct ✅ → File B incorrect ❌")
    print(f"  - File A incorrect ❌ → File B correct ✅")
    print(f"  - Both correct ✅✅")
    print(f"  - Both incorrect ❌❌")


if __name__ == "__main__":
    main()


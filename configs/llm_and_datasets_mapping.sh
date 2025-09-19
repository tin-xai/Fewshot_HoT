#!/bin/bash

# Define llm_models_dict
declare -A llm_models_dict=(
    ["1"]="gemini-1.5-flash-002"
    ["2"]="gemini-1.5-pro-002"
    ["3"]="gpt-4o-mini-2024-07-18"
    ["4"]="gpt-4o-2024-08-06"
    ["5"]="claude"
    ["6"]="llama_sambanova_8b"
    ["7"]="llama_sambanova_70b"
    ["8"]="llama_sambanova_405b"
    ["9"]="qwen25_coder_32b"
    ["10"]="qwq_32b"
    ["11"]="deepseek_r1"
    ["12"]="gemini_thinking"
    ["13"]="nebius_llama70b"
    ["14"]="nebius_llama405b"
)

# Define datasets_dict
declare -A datasets_dict=(
    ["1"]='GSM8K'
    ["2"]='MultiArith'
    ["3"]='ASDiv'
    ["4"]='SVAMP'
    ["5"]='AQUA'
    ["6"]='date'
    ["7"]='p_GSM8K'
    ["8"]='GSM_Plus'
    ["9"]='GSM_IC'
    ["10"]='GSM8K_Hard'
    ["11"]='StrategyQA'
    ["12"]='commonsenseQA'
    ["13"]='wikimultihopQA'
    ["14"]='sports'
    ["15"]='reclor'
    ["16"]='CLUTRR'
    ["17"]='object_counting'
    ["18"]='navigate'
    ["19"]='causal_judgement'
    ["20"]='logical_deduction_three_objects'
    ["21"]='logical_deduction_five_objects'
    ["22"]='logical_deduction_seven_objects'
    ["23"]='reasoning_about_colored_objects'
    ["24"]='spartQA'
    ["25"]='last_letter_2'
    ["26"]='last_letter_4'
    ["27"]='coin'
    ["28"]='word_sorting'
    ["29"]='tracking_shuffled_objects_seven_objects'
    ["30"]='gpqa'
    ["31"]='web_of_lies'
    ["32"]='temporal_sequences'
    ["33"]='drop_break'
    ["34"]='drop_cencus'
    ["35"]='squad'
    ["36"]='medQA'
    ["37"]='GSM_Symbolic'
    ["38"]='LIMO'
    ['39']='bbeh_causal_judgement'
    ['40']='bbeh_spatial_reasoning'
    ['41']='bbeh_object_attribute'
    ['42']='bbeh_boardgame'
    ['43']='bbeh_time_arithmetic'
    ['44']='bbeh_disambiguation'
    ['45']='bbeh_shuffle_objects'
    ['46']='rag4rag'
    ['47']='IFBench'
    
    
    
)

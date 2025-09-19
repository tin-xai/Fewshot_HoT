import re

def extract_last_sentence(question, dataset):
    if dataset in ['commonsenseQA']:
        pattern = re.compile(r"Question:\s*(.*?)\s*([^.?!]*[.?!])\s*Answer Choices:", re.DOTALL)
        match = pattern.search(question)
        return match.group(2) if match else 'the question'
    elif dataset == 'sports':
        return 'Is the following sentence plausible?'
    elif dataset == 'spartQA':
        return ""
    else:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        # Return the last sentence after stripping any extra spaces.
        return sentences[-1].strip() if sentences else text    

def remove_fact_tags_from_answers(prompt):
    return re.sub(
        r"Answer:.*?(<fact\d+>.*?</fact\d+>.*?)(?=\n|$)", 
        lambda match: "Answer: " + re.sub(r"</?fact\d+>", "", match.group(1)), 
        prompt, 
        flags=re.DOTALL
    )

def remove_fact_tags_from_questions(prompt):
    return re.sub(
        r"Reformatted Question:.*?(<fact\d+>.*?</fact\d+>.*?)(?=\n|$)", 
        lambda match: "Reformatted Question: " + re.sub(r"</?fact\d+>", "", match.group(1)), 
        prompt, 
        flags=re.DOTALL
    )
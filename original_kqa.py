import torch
import argparse
import json
import pickle
import numpy as np
from collections import Counter
from tqdm.contrib import tzip
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(description="Process model configurations for LLM evaluation.")
    parser.add_argument("--device_respond", default='cuda:3', help="Device for responding")
    parser.add_argument("--test", default=0, help="Test")
    return parser.parse_args()

def ask(prompt):   
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample = False
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def ask_sample(prompt):   
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample = True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def read_questions_from_jsonl(file_path):
    questions = [] 
    with open(file_path, 'r', encoding='utf-8') as file: 
        for line in file: 
            data = json.loads(line) 
            questions.append(data["Question"])  
    return questions  

def most_frequent(entail_result_temp):
    if len(entail_result_temp) == 0:
        return 0
    count = Counter(entail_result_temp)
    max_count = max(count.values())
    frequent_elements = [key for key, value in count.items() if value == max_count]
    if len(frequent_elements) > 1:
        return 1
    return frequent_elements[0]

def is_entails(question, llm_answer, answer):

    with open(f"datasets/K-QA/prompts/is_entails.txt", "r") as f:
        prompt_template_entails = f.read()

    prompt = prompt_template_entails.replace('{question}', question).replace('{llm_answer}', llm_answer).replace('{answer}', answer)
    answer = ask(prompt)
    if 'answer is False' in answer or 'answer is false' in answer or 'the answer as False' in answer:
        return 0
    elif 'answer is True' in answer or 'answer is true' in answer:
        return 1
    else:
        entail_result_temp = []
        for i in range(11):
            answer = ask_sample(prompt)
            if 'answer is False' in answer or 'answer is false' in answer or 'the answer as False' in answer:
                entail_result_temp.append(0)
            elif 'answer is True' in answer or 'answer is true' in answer:
                entail_result_temp.append(1)
            if len(entail_result_temp) == 3:
                break
        return most_frequent(entail_result_temp)

def is_contradict(question, llm_answer, answer):

    with open(f"datasets/K-QA/prompts/is_contradict.txt", "r") as f:
        prompt_template_contradict = f.read()

    prompt = prompt_template_contradict.replace('{question}', question).replace('{llm_answer}', llm_answer).replace('{answer}', answer)
    answer = ask(prompt)
    if 'answer is False' in answer or 'answer is false' in answer or 'the answer as False' in answer:
        return 0
    elif 'answer is True' in answer or 'answer is true' in answer:
        return 1
    else:
        entail_result_temp = []
        for i in range(11):
            answer = ask_sample(prompt)
            if 'answer is False' in answer or 'answer is false' in answer or 'the answer as False' in answer:
                entail_result_temp.append(0)
            elif 'answer is True' in answer or 'answer is true' in answer:
                entail_result_temp.append(1)
            if len(entail_result_temp) == 3:
                break
        return most_frequent(entail_result_temp)
    
def get_score(question_list, answer_list, mh_list):
        
    evaluate_entails = []
    for question, llm_answer, mh in tzip(question_list, answer_list, mh_list):
        result_single = []
        for answer in mh:
            result_single.append(is_entails(question, llm_answer, answer))
        evaluate_entails.append(np.array(result_single).sum() / len(mh))

    evaluate_contradict = []
    for question, llm_answer, mh in tzip(question_list, answer_list, mh_list):
        result_single = []
        for answer in mh:
            result_single.append(is_contradict(question, llm_answer, answer))
        evaluate_contradict.append(np.array(result_single).sum())  

    return evaluate_entails, evaluate_contradict

def main():
    args = get_args()
    device_respond = args.device_respond
    test = args.test

    global tokenizer
    global model

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    )

    model.to(device_respond)


    np.random.seed(1024)
    indices = np.arange(201)
    test_indices = np.random.choice(indices, size=50, replace=False)
    test_indices = test_indices.tolist()

    file_path = 'datasets/K-QA/dataset/questions_w_answers.jsonl' 
    questions_list = read_questions_from_jsonl(file_path)

    mh_list = []
    with open(file_path, 'r', encoding='utf-8') as file: 
        for line in file: 
            data = json.loads(line)  
            mh_list.append(data['Must_have'])

    if int(test):
        mh_list = [mh_list[i] for i in [0,1]]
        questions_list = [questions_list[i] for i in [0,1]]

    answer_list = [ask(question) for question in questions_list]

    evaluate_entails, evaluate_contradict = get_score(questions_list, answer_list, mh_list)

    with open('data/original/answer_kqa_original.pkl', 'wb') as f:
        pickle.dump(answer_list, f)

    with open('data/original/entails_kqa_original.pkl', 'wb') as f:
        pickle.dump(evaluate_entails, f)

    with open('data/original/contradict_kqa_original.pkl', 'wb') as f:
        pickle.dump(evaluate_contradict, f)

if __name__ == "__main__":
    main()





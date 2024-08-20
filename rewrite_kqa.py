import torch
import argparse
import json
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm
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

def ask_sample_1(prompt):   
    
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
        temperature=1,
        top_p=0.999,
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

def rewrite_questions(question):
    rewrite_ask_list = []
    if int(test):
        times = 2
    else:
        times = 10000
    for i in range(times):
        try:
            a = ask_sample_1('Rewriting question to make it more understandable, just give me the rewritten question without any other word: ' + question)
            if a not in rewrite_ask_list:
                rewrite_ask_list.append(a)
                print(a)
            if len(rewrite_ask_list) == 100:
                break
        except:
            continue
    return rewrite_ask_list

def generate_rewrite(questions_list, mh_list, name):

    try:
        with open('data/rewrite/rewrite_kqa_100_{}.pkl'.format(name), 'rb') as f:
            rewrite_q = pickle.load(f)
    except:
        rewrite_q = []

    for q in tqdm(questions_list[len(rewrite_q):]):
        rewrite_q.append(rewrite_questions(q))
        with open('rewrite_kqa_100_{}.pkl'.format(name), 'wb') as f:
            pickle.dump(rewrite_q, f)

    with open('data/rewrite/rewrite_kqa_100_{}.pkl'.format(name), 'wb') as f:
        pickle.dump(rewrite_q, f)

    try:
        with open('data/rewrite/rewrite_kqa_100_answer_{}.pkl'.format(name), 'rb') as f:
            rewrite_a = pickle.load(f)
    except: 
        rewrite_a = []

    for q_list in tqdm(rewrite_q[len(rewrite_a):]):
        temp_rewrite_a = []
        for q in q_list:
            answer = ask(q)
            temp_rewrite_a.append(answer)
        rewrite_a.append(temp_rewrite_a)
        with open('data/rewrite/rewrite_kqa_100_answer_{}.pkl'.format(name), 'wb') as f:
            pickle.dump(rewrite_a, f)

    with open('data/rewrite/rewrite_kqa_100_answer_{}.pkl'.format(name), 'wb') as f:
        pickle.dump(rewrite_a, f)

    try:
        with open('data/rewrite/rewrite_kqa_100_score_{}.pkl'.format(name), 'rb') as f:
            rewrite_s = pickle.load(f)
    except: 
        rewrite_s = []

    for q_list, a_list in tzip(rewrite_q[len(rewrite_s):], rewrite_a[len(rewrite_s):]):
        rewrite_s.append(get_score(q_list, a_list, mh_list))
        with open('data/rewrite/rewrite_kqa_100_score_{}.pkl'.format(name), 'wb') as f:
            pickle.dump(rewrite_s, f)

    with open('data/rewrite/rewrite_kqa_100_score_{}.pkl'.format(name), 'wb') as f:
        pickle.dump(rewrite_s, f)

def main():

    global test
    global tokenizer
    global model

    args = get_args()
    device_respond = args.device_respond
    test = args.test

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
    train_indices = np.setdiff1d(indices, test_indices)
    val_indices = np.random.choice(train_indices, size=50, replace=False)
    train_indices = np.setdiff1d(train_indices, val_indices)

    file_path = 'datasets/K-QA/dataset/questions_w_answers.jsonl'  
    questions_list = read_questions_from_jsonl(file_path)
    mh_list = []
    with open(file_path, 'r', encoding='utf-8') as file: 
        for line in file:
            data = json.loads(line) 
            mh_list.append(data['Must_have'])

    questions_list = [questions_list[i] for i in train_indices]
    mh_list = [mh_list[i] for i in train_indices]

    if int(test):
        questions_list = [questions_list[i] for i in [0,1]]
        mh_list = [mh_list[i] for i in [0,1]]

    generate_rewrite(questions_list, mh_list, 'train')



    file_path = 'datasets/K-QA/dataset/questions_w_answers.jsonl'  
    questions_list = read_questions_from_jsonl(file_path)
    mh_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file: 
            data = json.loads(line) 
            mh_list.append(data['Must_have'])

    questions_list = [questions_list[i] for i in val_indices]
    mh_list = [mh_list[i] for i in val_indices]

    if int(test):
        questions_list = [questions_list[i] for i in [0,1]]
        mh_list = [mh_list[i] for i in [0,1]]

    generate_rewrite(questions_list, mh_list, 'val')


if __name__ == "__main__":
    main()





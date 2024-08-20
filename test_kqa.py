import torch
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from tqdm.contrib import tzip
from llms.llama3_8b import get_answers as get_answers_llama3_8b
from llms.mistral_7b import get_answers as get_answers_mistral_7b
from llms.zephyr_7b import get_answers as get_answers_zephyr_7b
from llms.gemma_7b import get_answers as get_answers_gemma_7b
from llms.chatgpt import get_answers as get_answers_chatgpt
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(description="Process model configurations for LLM evaluation.")
    parser.add_argument("--name", default='kqa_test', help="Experiment name")
    parser.add_argument("--is_rewrite", type=int, default=0, help="Flag to enable question rewriting")
    parser.add_argument("--add_prompt", default=None, help="Additional prompt to append to each question")
    parser.add_argument("--device_judge", default='cuda:1', help="Device for judging")
    parser.add_argument("--device_rewriter", default='cuda:2', help="Device for rewriting")
    parser.add_argument("--device_respond", default='cuda:3', help="Device for responding")
    parser.add_argument("--rewriter_ckpt", default=None, help="Checkpoint for the rewriter model")
    parser.add_argument("--test", default=0, help="Test")
    parser.add_argument("--openai_api_key", default=None, help="Openai_api_key")
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

def ask_rewriter(prompt):   
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(rewriter.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = rewriter.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample = False
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

def rewrite_question(question):
    a = ask_rewriter('Rewriting question to make it more understandable, just give me the rewritten question without any other word: ' + question)
    return a

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

    return np.array(evaluate_entails).mean(), np.array(evaluate_contradict).mean()

def save_result(score_list, llm_list_finish, name):
    columns = ['comp', 'cont'] 
    df = pd.DataFrame(score_list, columns=columns)
    df = pd.DataFrame(score_list)
    df.index = llm_list_finish
    df.to_csv('test_result/' + name + '.csv')

def main():
    args = get_args()
    is_rewrite = args.is_rewrite
    add_prompt = args.add_prompt
    device_judge = args.device_judge 
    device_rewriter = args.device_rewriter
    device_respond = args.device_respond
    rewriter_ckpt = args.rewriter_ckpt
    test = args.test
    openai_api_key = args.openai_api_key

    global tokenizer
    global model

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    )
    print(device_judge)
    model.to(device_judge)


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

    mh_list = [mh_list[i] for i in test_indices]
    questions_list = [questions_list[i] for i in test_indices]

    if int(test):
        mh_list = [mh_list[i] for i in [0,1]]
        questions_list = [questions_list[i] for i in [0,1]]

    if is_rewrite:

        global rewriter
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        custom_weights_path = rewriter_ckpt
        device = device_rewriter
        rewriter = AutoModelForCausalLM.from_pretrained(model_id)
        custom_state_dict = torch.load(custom_weights_path, map_location="cpu")
        rewriter.load_state_dict(custom_state_dict['state'])
        rewriter = rewriter.to(dtype=torch.bfloat16)
        rewriter.to(device)

        rewrite_q = []

        for q in tqdm(questions_list[len(rewrite_q):]):
            rewrite_q.append(rewrite_question(q))

    else:
        rewrite_q = questions_list[:]

    llm_list = ['llama3_8b', 'mistral_7b', 'zephyr_7b', 'gemma_7b', 'gpt35', 'gpt4o']
    #llm_list = ['llama3_8b', 'mistral_7b', 'zephyr_7b', 'gemma_7b', 'gpt35', 'gpt4o']

    try:
        df = pd.read_csv('test_result/' + args.name + '.csv', index_col=0)
        llm_list_finish = df.index.tolist()
        score_list = df.values.tolist()
    except:
        llm_list_finish = []
        score_list = []

    for llm in llm_list:
        with torch.cuda.device(device_respond):
            torch.cuda.empty_cache()

        if llm == 'llama3_8b' and 'llama3_8b' not in llm_list_finish:
            if add_prompt:
                rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                rewrite_a = get_answers_llama3_8b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_llama3_8b(device_respond, rewrite_q)
            score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            llm_list_finish.append('llama3_8b')
            save_result(score_list, llm_list_finish, args.name)
        if llm == 'mistral_7b' and 'mistral_7b' not in llm_list_finish:
            if add_prompt:
                rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                rewrite_a = get_answers_mistral_7b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_mistral_7b(device_respond, rewrite_q)
            score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            llm_list_finish.append('mistral_7b')
            save_result(score_list, llm_list_finish, args.name)
        if llm == 'zephyr_7b' and 'zephyr_7b' not in llm_list_finish:
            if add_prompt:
                rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                rewrite_a = get_answers_zephyr_7b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_zephyr_7b(device_respond, rewrite_q)
            score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            llm_list_finish.append('zephyr_7b')
            save_result(score_list, llm_list_finish, args.name)
        if llm == 'gemma_7b' and 'gemma_7b' not in llm_list_finish:
            if add_prompt:
                rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                rewrite_a = get_answers_gemma_7b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_gemma_7b(device_respond, rewrite_q)
            score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            llm_list_finish.append('gemma_7b')
            save_result(score_list, llm_list_finish, args.name)
        if llm == 'gpt35' and 'gpt35' not in llm_list_finish:
            try:
                if add_prompt:
                    rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-3.5-turbo-1106', rewrite_q_prompt)
                else:
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-3.5-turbo-1106', rewrite_q)
                score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            except:
                score_list.append([-100, -100])
            llm_list_finish.append('gpt35')
            save_result(score_list, llm_list_finish, args.name)
        if llm == 'gpt4o' and 'gpt4o' not in llm_list_finish:
            try:
                if add_prompt:
                    rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-4o-2024-05-13', rewrite_q_prompt)
                else:
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-4o-2024-05-13', rewrite_q)
                score_list.append([i for i in get_score(rewrite_q, rewrite_a, mh_list)])
            except:
                score_list.append([-100, -100])
            llm_list_finish.append('gpt4o')
            save_result(score_list, llm_list_finish, args.name)

if __name__ == "__main__":
    main()





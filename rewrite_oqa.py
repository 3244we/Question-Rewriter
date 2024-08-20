import torch
import argparse
import pickle
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tzip
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

def get_args():
    parser = argparse.ArgumentParser(description="Process model configurations for LLM evaluation.")
    parser.add_argument("--device_judge", default='cuda:1', help="Device for judging")
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
        temperature=1,
        top_p=0.999,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)
    
def get_score(question_list, answer_list):       
    reward = []
    with torch.no_grad():
        for prompt, chosen in tqdm(zip(question_list, answer_list)):
            reward.append(float(torch.sigmoid(model_reward(**tokenizer_reward("prompter: {} assistant: {}".format(prompt, chosen), return_tensors='pt')).logits)[0][0]))
    return reward

def rewrite_questions(question):
    rewrite_ask_list = []
    if int(test):
        times = 2
    else:
        times = 10000
    for i in range(times):
        try:
            a = ask_sample('Rewriting question to make it more understandable, just give me the rewritten question without any other word: ' + question)
            if a not in rewrite_ask_list:
                rewrite_ask_list.append(a)
                print(a)
            if len(rewrite_ask_list) == 100:
                break
        except:
            continue
    return rewrite_ask_list

def generate_rewrite(questions_list, name):

    try:
        with open('data/rewrite/rewrite_oqa_100_{}.pkl'.format(name), 'rb') as f:
            rewrite_q = pickle.load(f)
    except:
        rewrite_q = []

    for q in tqdm(questions_list[len(rewrite_q):]):
        rewrite_q.append(rewrite_questions(q))
        with open('rewrite_oqa_100_{}.pkl'.format(name), 'wb') as f:
            pickle.dump(rewrite_q, f)

    with open('data/rewrite/rewrite_oqa_100_{}.pkl'.format(name), 'wb') as f:
        pickle.dump(rewrite_q, f)

    try:
        with open('data/rewrite/rewrite_oqa_100_answer_{}.pkl'.format(name), 'rb') as f:
            rewrite_a = pickle.load(f)
    except: 
        rewrite_a = []

    for q_list in tqdm(rewrite_q[len(rewrite_a):]):
        temp_rewrite_a = []
        for q in q_list:
            answer = ask(q)
            temp_rewrite_a.append(answer)
        rewrite_a.append(temp_rewrite_a)
        with open('data/rewrite/rewrite_oqa_100_answer_{}.pkl'.format(name), 'wb') as f:
            pickle.dump(rewrite_a, f)

    with open('data/rewrite/rewrite_oqa_100_answer_{}.pkl'.format(name), 'wb') as f:
        pickle.dump(rewrite_a, f)

    try:
        with open('data/rewrite/rewrite_oqa_100_score_{}.pkl'.format(name), 'rb') as f:
            rewrite_s = pickle.load(f)
    except: 
        rewrite_s = []

    for q_list, a_list in tzip(rewrite_q[len(rewrite_s):], rewrite_a[len(rewrite_s):]):
        rewrite_s.append(get_score(q_list, a_list))
        with open('data/rewrite/rewrite_oqa_100_score_{}.pkl'.format(name), 'wb') as f:
            pickle.dump(rewrite_s, f)

    with open('data/rewrite/rewrite_oqa_100_score_{}.pkl'.format(name), 'wb') as f:
        pickle.dump(rewrite_s, f)

def main():

    global test
    global tokenizer_reward
    global model_reward
    global tokenizer
    global model


    args = get_args()
    device_judge = args.device_judge 
    device_respond = args.device_respond
    test = args.test


    with torch.cuda.device(device_judge):
        peft_model_id = "vincentmin/llama-2-7b-reward-oasst1"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=1,
            load_in_4bit=True,
            torch_dtype=torch.float16,
        )
        model_reward = PeftModel.from_pretrained(model, peft_model_id)
        tokenizer_reward = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_auth_token=True)

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    )

    model.to(device_respond)


    questions_list = pd.read_csv('datasets/OQA/oasst1_train.csv')['prompt'].tolist()

    if int(test):
        questions_list = [questions_list[i] for i in [0,1]]

    generate_rewrite(questions_list, 'train')



    questions_list = pd.read_csv('datasets/OQA/oasst1_val.csv')['prompt'].tolist()

    if int(test):
        questions_list = [questions_list[i] for i in [0,1]]

    generate_rewrite(questions_list, 'val')




if __name__ == "__main__":
    main()





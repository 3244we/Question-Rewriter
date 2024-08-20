import pickle
import json
import random
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

def generate_answer_pairs(answers, scores, its):

    random.seed(1024)
    
    if len(answers) == 1:
        return []
        
    # Pair the answers with their scores
    paired_answers = list(zip(answers, scores))
    
    # Separate answers into max_half and min_half based on the threshold 'its'
    max_half = [ans for ans in paired_answers if ans[1] > its]
    min_half = [ans for ans in paired_answers if ans[1] < its]
    
    # Sort max_half by score descending
    max_half_sorted = sorted(max_half, key=lambda x: x[1], reverse=True)
    
    # Select the top 5 answers from max_half
    selected_max_half = max_half_sorted[:4]
    
    # Randomly select 5 answers from min_half
    selected_min_half = random.sample(min_half, min(5, len(min_half)))  # Ensure not to sample more than the available answers

    # Combine the selected answers from both halves
    result_pairs = [(a[0], b[0]) for a in selected_max_half for b in selected_min_half]

    return result_pairs



def create_json_data(prompts, response_pairs):
    data = []
    for prompt, (good_response, bad_response) in zip(prompts, response_pairs):
        entry = {
            "prompt": prompt,
            "good_response": good_response,
            "bad_response": bad_response
        }
        data.append(entry)
    
    json_data = json.dumps(data, indent=4)
    return json_data

with open('data/rewrite/rewrite_oqa_100_train.pkl', 'rb') as f:
    train_rewrite_q = pickle.load(f)

with open('data/rewrite/rewrite_oqa_100_val.pkl', 'rb') as f:
    val_rewrite_q = pickle.load(f)

with open('data/rewrite/rewrite_oqa_100_score_train.pkl', 'rb') as f:
    train_rewrite_s = pickle.load(f)

with open('data/rewrite/rewrite_oqa_100_score_val.pkl', 'rb') as f:
    val_rewrite_s = pickle.load(f)

with open('data/original/score_oqa_train_original.pkl', 'rb') as f:
    train_initial_s = pickle.load(f)

with open('data/original/score_oqa_val_original.pkl', 'rb') as f:
    val_initial_s = pickle.load(f)

data = pd.read_csv('datasets/OQA/oasst1_train.csv')
question_list_train = data['prompt'].tolist()

data = pd.read_csv('datasets/OQA/oasst1_val.csv')
question_list_val = data['prompt'].tolist()

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)


train_prompt_list = []

for i in question_list_train:
    messages = [
        {"role": "user", "content": 'Rewriting question to make it more understandable, just give me the rewritten question without any other word: ' + i}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    prompt = tokenizer.decode(input_ids[0])
    train_prompt_list.append(prompt)


val_prompt_list = []
for i in question_list_val:
    messages = [
        {"role": "user", "content": 'Rewriting question to make it more understandable, just give me the rewritten question without any other word: ' + i}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    prompt = tokenizer.decode(input_ids[0])
    val_prompt_list.append(prompt)


#train_indices = [0,1]
#val_indices = [0,1]


response_pairs = []
prompts = []
for pp, i, j, its in zip(train_prompt_list, train_rewrite_q, train_rewrite_s, train_initial_s): #for pp, i, j, its in zip(train_prompt_list, train_rewrite_q, train_rewrite_s, train_initial_s):
    pairs = generate_answer_pairs(i, j, its)
    if len(pairs) != 0:
        for p in pairs:
            if len(tokenizer.encode(pp + p[0])) <= 512 and len(tokenizer.encode(pp + p[1])) <= 512:#len(tokenizer.encode(pp + p[0])) <= 384 and len(tokenizer.encode(pp + p[1])) <= 384
                response_pairs.append(p)
                prompts.append(pp)

# Set random seed
random.seed(1024)

shuffle_index = list(range(len(response_pairs)))
# Shuffle both lists
response_pairs = np.array(response_pairs)[shuffle_index].tolist()
prompts = np.array(prompts)[shuffle_index].tolist()

json_data = create_json_data(prompts, response_pairs)
with open('data/dpo/dpo_train_data_20.json', 'w') as file:
    file.write(json_data)


response_pairs = []
prompts = []
for pp, i, j, its in zip(val_prompt_list, val_rewrite_q, val_rewrite_s, val_initial_s): #for pp, i, j, its in zip(train_prompt_list, train_rewrite_q, train_rewrite_s, train_initial_s):
    pairs = generate_answer_pairs(i, j, its)
    if len(pairs) != 0:
        for p in pairs:
            if len(tokenizer.encode(pp + p[0])) <= 512 and len(tokenizer.encode(pp + p[1])) <= 512:#len(tokenizer.encode(pp)) <= 192 and len(tokenizer.encode(pp + p[0])) <= 384 and len(tokenizer.encode(pp + p[1])) <= 384
                response_pairs.append(p)
                prompts.append(pp)

# Set random seed
random.seed(1024)

shuffle_index = list(range(len(response_pairs)))
# Shuffle both lists
response_pairs = np.array(response_pairs)[shuffle_index].tolist()
prompts = np.array(prompts)[shuffle_index].tolist()

shuffle_index = list(range(len(response_pairs)))
# Shuffle both lists
response_pairs = np.array(response_pairs)[shuffle_index].tolist()
prompts = np.array(prompts)[shuffle_index].tolist()

json_data = create_json_data(prompts, response_pairs)
with open('data/dpo/oqa_eval_data_20.json', 'w') as file:
    file.write(json_data)

print('size of val_set: ' + str(len(response_pairs)))
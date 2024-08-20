import pickle
import json
import random
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

def generate_answer_pairs(answers, scores, its, s1, s2, s3, s4):

    random.seed(1024)
    
    if len(answers) == 1:
        return []
        
    # Pair the answers with their scores
    paired_answers = list(zip(answers, scores, s1, s2))
    
    # Separate answers into max_half and min_half based on the threshold 'its'
    max_half = [ans for ans in paired_answers if ans[2] > s3 and ans[3] > s4 and (ans[2] > s3 or ans[3] > s4)]
    min_half = [ans for ans in paired_answers if ans[2] < s3 and ans[3] < s4 and (ans[2] < s3 or ans[3] < s4)]
    
    # Sort max_half by score descending
    max_half_sorted = sorted(max_half, key=lambda x: (x[1], x[3]))
    
    # Select the top 5 answers from max_half
    selected_max_half = max_half_sorted[:5]
    
    # Randomly select 5 answers from min_half
    selected_min_half = random.sample(min_half, min(10, len(min_half)))  # Ensure not to sample more than the available answers

    # Combine the selected answers from both halves
    result_pairs = [(a[0], b[0]) for a in selected_max_half for b in selected_min_half]

    return result_pairs

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    return df

def create_json_data(prompts, response_pairs):
    data = []
    for prompt, (good_response, bad_response) in zip(prompts, response_pairs):
        entry = {
            "prompt": prompt,
            "good_response": good_response,
            "bad_response": bad_response
        }
        data.append(entry)
    
    # 转换成JSON格式的字符串
    json_data = json.dumps(data, indent=4)
    return json_data

def instruct_prompt(x):

    prompt_list = []
    for i in x['prompt']:
        
        messages = [
        {"role": "user", "content": i}
        ]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        prompt = tokenizer.decode(input_ids[0])
        
        prompt_list.append(prompt)

    return prompt_list

with open('data/rewrite/rewrite_tqa_100_train.pkl', 'rb') as f:
    train_rewrite_q = pickle.load(f)

with open('data/rewrite/rewrite_tqa_100_val.pkl', 'rb') as f:
    val_rewrite_q = pickle.load(f)

with open('data/rewrite/rewrite_tqa_100_score_train.pkl', 'rb') as f:
    train_rewrite_s = pickle.load(f)

with open('data/rewrite/rewrite_tqa_100_score_val.pkl', 'rb') as f:
    val_rewrite_s = pickle.load(f)

with open('data/original/ts_tqa_original.pkl', 'rb') as f:
    initial_is_ts = pickle.load(f)

with open('data/original/is_tqa_original.pkl', 'rb') as f:
    initial_is_cs = pickle.load(f)

with open('data/original/as_tqa_original.pkl', 'rb') as f:
    initial_s = pickle.load(f)

train_rewrite_ies = [i[0] for i in train_rewrite_s]
train_rewrite_ics = [i[1] for i in train_rewrite_s]
val_rewrite_ies = [i[0] for i in val_rewrite_s]
val_rewrite_ics = [i[1] for i in val_rewrite_s]
train_rewrite_s = [i[2] for i in train_rewrite_s]
train_rewrite_s = [i[2] for i in val_rewrite_s]

file_path = 'datasets/TruthfulQA/data/finetune_truth.jsonl'
df = read_jsonl(file_path)
df['completion'] = [i.replace(' ', '') for i in df['completion']]
df['question'] = [df['prompt'][i].split('\nA:')[0] for i in range(len(df))]

np.random.seed(1024)
prompts = df['question'].unique()
prompts.sort()

train_prompts = np.random.choice(prompts, size=int(len(prompts) * 0.75), replace=False)
eval_prompts = np.random.choice(train_prompts, size=int(len(prompts) * 0.25), replace=False)
train_prompts = np.setdiff1d(train_prompts, eval_prompts)

all_prompts = list(prompts)
train_indices = [all_prompts.index(prompt) for prompt in train_prompts]
val_indices = [all_prompts.index(prompt) for prompt in eval_prompts]

question_list = [i.split('Q: ')[1] for i in prompts.tolist()]

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt_list = []
for i in question_list:
    
    messages = [
    {"role": "user", "content": 'Rewriting question to make it more understandable, just give me the rewritten question without any other word: ' + i}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    prompt = tokenizer.decode(input_ids[0])
    
    prompt_list.append(prompt)

# Use the indices to create lists for different sets from prompt_list
train_prompt_list = [prompt_list[i] for i in train_indices]
val_prompt_list = [prompt_list[i] for i in val_indices]


#train_indices = [0,1]
#val_indices = [0,1]

train_initial_s = [initial_s[i] for i in train_indices]
val_initial_s = [initial_s[i] for i in val_indices]
train_initial_is_ts = [initial_is_ts[i] for i in train_indices]
val_initial_is_ts = [initial_is_ts[i] for i in val_indices]
train_initial_is_cs = [initial_is_cs[i] for i in train_indices]
val_initial_is_cs = [initial_is_cs[i] for i in val_indices]



response_pairs = []
prompts = []
for pp, i, j, its, s1, s2, s3, s4 in zip(train_prompt_list, train_rewrite_q, train_rewrite_s, train_initial_s, train_rewrite_ies, train_rewrite_ics, train_initial_is_ts, train_initial_is_cs):
    pairs = generate_answer_pairs(i, j, its, s1, s2, s3, s4)
    if len(pairs) != 0:
        for p in pairs:
            response_pairs.append(p)
            prompts.append(pp)

random.seed(1024)
shuffle_index = list(range(len(response_pairs)))
response_pairs = np.array(response_pairs)[shuffle_index].tolist()
prompts = np.array(prompts)[shuffle_index].tolist()

json_data = create_json_data(prompts, response_pairs)
with open('data/dpo/tqa_train_data_50.json', 'w') as file:
    file.write(json_data)


response_pairs = []
prompts = []
for pp, i, j, its, s1, s2, s3, s4 in zip(val_prompt_list, val_rewrite_q, val_rewrite_s, val_initial_s, val_rewrite_ies, val_rewrite_ics, val_initial_is_ts, val_initial_is_cs):
    pairs = generate_answer_pairs(i, j, its, s1, s2, s3, s4)
    if len(pairs) != 0:
        for p in pairs:
            response_pairs.append(p)
            prompts.append(pp)

shuffle_index = list(range(len(response_pairs)))
# Shuffle both lists
response_pairs = np.array(response_pairs)[shuffle_index].tolist()
prompts = np.array(prompts)[shuffle_index].tolist()

json_data = create_json_data(prompts, response_pairs)
with open('data/dpo/tqa_eval_data_50.json', 'w') as file:
    file.write(json_data)

print('size of val_set: ' + str(len(response_pairs)))
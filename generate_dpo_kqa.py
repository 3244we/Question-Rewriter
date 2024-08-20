import pickle
import json
import random
import numpy as np
from transformers import AutoTokenizer

def generate_answer_pairs_random(answers, scores, its, s1, s2, s3, s4):

    random.seed(1024)
    
    if len(answers) == 1:
        return []
        
    # Pair the answers with their scores
    paired_answers = list(zip(answers, scores, s1, s2))
    
    # Separate answers into max_half and min_half based on the threshold 'its'
    max_half = [ans for ans in paired_answers if ans[2] >= s3 and ans[3] <= s4 and (ans[2] > s3 or ans[3] < s4)]
    min_half = [ans for ans in paired_answers if ans[2] <= s3 and ans[3] >= s4 and (ans[2] < s3 or ans[3] > s4)]
    
    # Select the top 5 answers from max_half
    selected_max_half = random.sample(max_half, min(10, len(max_half)))
    
    # Randomly select 5 answers from min_half
    selected_min_half = random.sample(min_half, min(20, len(min_half)))  # Ensure not to sample more than the available answers

    # Combine the selected answers from both halves
    result_pairs = [(a[0], b[0]) for a in selected_max_half for b in selected_min_half]

    return result_pairs

def generate_answer_pairs_extreme(answers, scores, its, s1, s2, s3, s4):

    random.seed(1024)
    
    if len(answers) == 1:
        return []
        
    # Pair the answers with their scores
    paired_answers = list(zip(answers, scores, s1, s2))
    
    # Separate answers into max_half and min_half based on the threshold 'its'
    max_half = [ans for ans in paired_answers if ans[2] >= s3 and ans[3] <= s4 and (ans[2] > s3 or ans[3] < s4)]
    min_half = [ans for ans in paired_answers if ans[2] <= s3 and ans[3] >= s4 and (ans[2] < s3 or ans[3] > s4)]
    
    # Sort max_half by score descending
    max_half_sorted = sorted(max_half, key=lambda x: (-x[3], x[2]))
    min_half_sorted = sorted(min_half, key=lambda x: (x[3], -x[2]))
    
    # Select the top 5 answers from max_half
    selected_max_half = max_half_sorted[:10]
    
    # Randomly select 5 answers from min_half
    selected_min_half = min_half_sorted[:20]

    # Combine the selected answers from both halves
    result_pairs = [(a[0], b[0]) for a in selected_max_half for b in selected_min_half]

    return result_pairs

def generate_answer_pairs_inverse(answers, scores, its, s1, s2, s3, s4):

    random.seed(1024)
    
    if len(answers) == 1:
        return []
        
    # Pair the answers with their scores
    paired_answers = list(zip(answers, scores, s1, s2))
    
    # Separate answers into max_half and min_half
    max_half = [ans for ans in paired_answers if ans[2] >= s3 and ans[3] <= s4 and (ans[2] > s3 or ans[3] < s4)]
    min_half = [ans for ans in paired_answers if ans[2] <= s3 and ans[3] >= s4 and (ans[2] < s3 or ans[3] > s4)]
    
    # Sort max_half by score descending
    min_half_sorted = sorted(min_half, key=lambda x: (x[3], -x[2]))
    
    # Select the top 5 answers from max_half
    selected_max_half = random.sample(max_half, min(10, len(max_half)))
    
    # Randomly select 5 answers from min_half
    selected_min_half = min_half_sorted[:20]

    # Combine the selected answers from both halves
    result_pairs = [(a[0], b[0]) for a in selected_max_half for b in selected_min_half]

    return result_pairs

def generate_answer_pairs(answers, scores, its, s1, s2, s3, s4):

    random.seed(1024)
    
    if len(answers) == 1:
        return []
        
    # Pair the answers with their scores
    paired_answers = list(zip(answers, scores, s1, s2))
    
    # Separate answers into max_half and min_half
    max_half = [ans for ans in paired_answers if ans[2] >= s3 and ans[3] <= s4 and (ans[2] > s3 or ans[3] < s4)]
    min_half = [ans for ans in paired_answers if ans[2] <= s3 and ans[3] >= s4 and (ans[2] < s3 or ans[3] > s4)]
    
    # Sort max_half by score descending
    max_half_sorted = sorted(max_half, key=lambda x: (-x[3], x[2]))
    
    # Select the top 5 answers from max_half
    selected_max_half = max_half_sorted[:10]
    
    # Randomly select 5 answers from min_half
    selected_min_half = random.sample(min_half, min(20, len(min_half)))  # Ensure not to sample more than the available answers

    # Combine the selected answers from both halves
    result_pairs = [(a[0], b[0]) for a in selected_max_half for b in selected_min_half]

    return result_pairs

def read_questions_from_jsonl(file_path):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            questions.append(data["Question"])
    return questions

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

with open('data/rewrite/rewrite_kqa_100_train.pkl', 'rb') as f:
    train_rewrite_q = pickle.load(f)

with open('data/rewrite/rewrite_kqa_100_val.pkl', 'rb') as f:
    val_rewrite_q = pickle.load(f)

with open('data/rewrite/rewrite_kqa_100_score_train.pkl', 'rb') as f:
    train_rewrite_s = pickle.load(f)

with open('data/rewrite/rewrite_kqa_100_score_val.pkl', 'rb') as f:
    val_rewrite_s = pickle.load(f)

with open('data/original/entails_kqa_original.pkl', 'rb') as f:
    initial_is_entails = pickle.load(f)

with open('data/original/contradict_kqa_original.pkl', 'rb') as f:
    initial_is_contradict = pickle.load(f)

train_rewrite_ies = [i[0] for i in train_rewrite_s]
train_rewrite_ics = [i[1] for i in train_rewrite_s]
val_rewrite_ies = [i[0] for i in val_rewrite_s]
val_rewrite_ics = [i[1] for i in val_rewrite_s]

train_rewrite_s = np.array(train_rewrite_ies) - 0.5 * np.array(train_rewrite_ics) # not useful
train_rewrite_s = np.array(val_rewrite_ies) - 0.5 * np.array(val_rewrite_ics) # not useful

initial_s = np.array(initial_is_entails) - np.array(initial_is_contradict)

file_path = 'datasets/K-QA/dataset/questions_w_answers.jsonl'
question_list = read_questions_from_jsonl(file_path)

np.random.seed(1024)
indices = np.arange(len(question_list))
test_indices = np.random.choice(indices, size=50, replace=False)
train_indices = np.setdiff1d(indices, test_indices)
val_indices = np.random.choice(train_indices, size=50, replace=False)
train_indices = np.setdiff1d(train_indices, val_indices)

test_indices = test_indices.tolist()
train_indices = train_indices.tolist()
val_indices = val_indices.tolist()

question_list_train = [question_list[i] for i in train_indices]
question_list_val = [question_list[i] for i in val_indices]

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

train_initial_s = [initial_s[i] for i in train_indices]
val_initial_s = [initial_s[i] for i in val_indices]
train_initial_is_entails = [initial_is_entails[i] for i in train_indices]
val_initial_is_entails = [initial_is_entails[i] for i in val_indices]
train_initial_is_contradict = [initial_is_contradict[i] for i in train_indices]
val_initial_is_contradict = [initial_is_contradict[i] for i in val_indices]



response_pairs = []
prompts = []
for pp, i, j, its, s1, s2, s3, s4 in zip(train_prompt_list, train_rewrite_q, train_rewrite_s, train_initial_s, train_rewrite_ies, train_rewrite_ics, train_initial_is_entails, train_initial_is_contradict):
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
with open('data/dpo/kqa_train_data_200.json', 'w') as file:
    file.write(json_data)


response_pairs = []
prompts = []
for pp, i, j, its, s1, s2, s3, s4 in zip(val_prompt_list, val_rewrite_q, val_rewrite_s, val_initial_s, val_rewrite_ies, val_rewrite_ics, val_initial_is_entails, val_initial_is_contradict):
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
with open('data/dpo/kqa_eval_data_200.json', 'w') as file:
    file.write(json_data)

print('size of val_set: ' + str(len(response_pairs)))
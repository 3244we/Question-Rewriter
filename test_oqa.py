import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from llms.llama3_8b import get_answers as get_answers_llama3_8b
from llms.mistral_7b import get_answers as get_answers_mistral_7b
from llms.zephyr_7b import get_answers as get_answers_zephyr_7b
from llms.gemma_7b import get_answers as get_answers_gemma_7b
from llms.chatgpt import get_answers as get_answers_chatgpt
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification



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


def rewrite_question(question):
    a = ask_rewriter('Rewriting question to make it more understandable, just give me the rewritten question without any other word: ' + question)
    return a

    
def get_score(question_list, answer_list):       
    reward = []
    with torch.no_grad():
        for prompt, chosen in tqdm(zip(question_list, answer_list)):
            reward.append(float(torch.sigmoid(reward_model(**tokenizer("prompter: {} assistant: {}".format(prompt, chosen), return_tensors='pt')).logits)[0][0]))
    return np.array(reward).mean()

def save_result(score_list, llm_list_finish, name):
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
    global reward_model

    with torch.cuda.device(device_judge):
        peft_model_id = "vincentmin/llama-2-7b-reward-oasst1"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=1,
            load_in_4bit=True,
            torch_dtype=torch.float16,
        )
        reward_model = PeftModel.from_pretrained(model, peft_model_id)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_auth_token=True)

    questions_list = pd.read_csv('datasets/OQA/oasst1_test.csv')['prompt'].tolist()

    if int(test):
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
            score_list.append([get_score(rewrite_q, rewrite_a)])
            llm_list_finish.append('llama3_8b')
            save_result(score_list, llm_list_finish, args.name)
        if llm == 'mistral_7b' and 'mistral_7b' not in llm_list_finish:
            if add_prompt:
                rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                rewrite_a = get_answers_mistral_7b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_mistral_7b(device_respond, rewrite_q)
            score_list.append([get_score(rewrite_q, rewrite_a)])
            llm_list_finish.append('mistral_7b')
            save_result(score_list, llm_list_finish, args.name)
        if llm == 'zephyr_7b' and 'zephyr_7b' not in llm_list_finish:
            if add_prompt:
                rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                rewrite_a = get_answers_zephyr_7b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_zephyr_7b(device_respond, rewrite_q)
            score_list.append([get_score(rewrite_q, rewrite_a)])
            llm_list_finish.append('zephyr_7b')
            save_result(score_list, llm_list_finish, args.name)
        if llm == 'gemma_7b' and 'gemma_7b' not in llm_list_finish:
            if add_prompt:
                rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                rewrite_a = get_answers_gemma_7b(device_respond, rewrite_q_prompt)
            else:
                rewrite_a = get_answers_gemma_7b(device_respond, rewrite_q)
            score_list.append([get_score(rewrite_q, rewrite_a)])
            llm_list_finish.append('gemma_7b')
            save_result(score_list, llm_list_finish, args.name)
        if llm == 'gpt35' and 'gpt35' not in llm_list_finish:
            try:
                if add_prompt:
                    rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-3.5-turbo-1106', rewrite_q_prompt)
                else:
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-3.5-turbo-1106', rewrite_q)
                score_list.append([get_score(rewrite_q, rewrite_a)])
            except:
                score_list.append([-100])
            llm_list_finish.append('gpt35')
            save_result(score_list, llm_list_finish, args.name)
        if llm == 'gpt4o' and 'gpt4o' not in llm_list_finish:
            try:
                if add_prompt:
                    rewrite_q_prompt = [i + add_prompt for i in rewrite_q]
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-4o-2024-05-13', rewrite_q_prompt)
                else:
                    rewrite_a = get_answers_chatgpt(openai_api_key, 'gpt-4o-2024-05-13', rewrite_q)
                score_list.append([get_score(rewrite_q, rewrite_a)])
            except:
                score_list.append([-100])
            llm_list_finish.append('gpt4o')
            save_result(score_list, llm_list_finish, args.name)

if __name__ == "__main__":
    main()





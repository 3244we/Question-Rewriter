import torch
import argparse
import pickle
import pandas as pd
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from llms.llama3_8b import get_answers as get_answers_llama3_8b
from transformers import AutoTokenizer, AutoModelForSequenceClassification



def get_args():
    parser = argparse.ArgumentParser(description="Process model configurations for LLM evaluation.")
    parser.add_argument("--device_judge", default='cuda:1', help="Device for judging")
    parser.add_argument("--device_respond", default='cuda:3', help="Device for responding")
    parser.add_argument("--test", default=0, help="Test")
    return parser.parse_args()

    
def get_score(question_list, answer_list):       
    reward = []
    with torch.no_grad():
        for prompt, chosen in tqdm(zip(question_list, answer_list)):
            reward.append(float(torch.sigmoid(reward_model(**tokenizer("prompter: {} assistant: {}".format(prompt, chosen), return_tensors='pt')).logits)[0][0]))
    return reward

def main():
    args = get_args()
    device_judge = args.device_judge 
    device_respond = args.device_respond
    test = args.test


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

    answer_list = get_answers_llama3_8b(device_respond, questions_list)
    score_list = get_score(questions_list, answer_list)

    with open('data/original/answer_oqa_test_original.pkl', 'wb') as f:
        pickle.dump(answer_list, f)

    with open('data/original/score_oqa_test_original.pkl', 'wb') as f:
        pickle.dump(score_list, f)



    with torch.cuda.device(device_respond):
        torch.cuda.empty_cache()

    questions_list = pd.read_csv('datasets/OQA/oasst1_train.csv')['prompt'].tolist()

    if int(test):
        questions_list = [questions_list[i] for i in [0,1]]

    answer_list = get_answers_llama3_8b(device_respond, questions_list)
    score_list = get_score(questions_list, answer_list)

    with open('data/original/answer_oqa_train_original.pkl', 'wb') as f:
        pickle.dump(answer_list, f)

    with open('data/original/score_oqa_train_original.pkl', 'wb') as f:
        pickle.dump(score_list, f)



    with torch.cuda.device(device_respond):
        torch.cuda.empty_cache()

    questions_list = pd.read_csv('datasets/OQA/oasst1_val.csv')['prompt'].tolist()

    if int(test):
        questions_list = [questions_list[i] for i in [0,1]]

    answer_list = get_answers_llama3_8b(device_respond, questions_list)
    score_list = get_score(questions_list, answer_list)

    with open('data/original/answer_oqa_val_original.pkl', 'wb') as f:
        pickle.dump(answer_list, f)
    with open('data/original/score_oqa_val_original.pkl', 'wb') as f:
        pickle.dump(score_list, f)

if __name__ == "__main__":
    main()





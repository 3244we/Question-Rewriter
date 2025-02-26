# Putting People in LLMs’ Shoes: Generating Better Answers via Question Rewriter

<p align="center">
<img src="https://github.com/3244we/Question-Rewriter/blob/master/img/pps_new.png" width="800">
</p>

This includes the code for generating data and training the rewriter using DPO. The DPO training code is derived from https://github.com/eric-mitchell/direct-preference-optimization with minor modifications. The model used for making judgments on TruthfulQA will be available for download after the publication of our paper, hence the code for generating data and testing the rewriter on TruthfulQA is temporarily unavailable. The version of cuda that we use is 11.8. [The paper is currently on arXiv.](https://arxiv.org/abs/2408.10573)

## Installation

Follow these steps to install the necessary dependencies and get the project up and running.

1. **Create a virtual environment:**

   ```bash
    conda create --name myenv python=3.8.0
   ```

2. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Obtain download permissions for llama_3 and log into your Hugging Face account in the system.**

## Setup

1. **Activate virtual environment:**

   ```bash
    conda activate myenv 
   ```

2. **Set the wandb token:**

   ```bash
    Replace the ??? in line 31 of direct-preference-optimization/train.py with your own wandb token. 
   ```
   
## K-QA

Training and testing rewriter on K-QA.

1. **Generate original answers and scores. The results are located in data/original/. Each model requires 17 GB of GPU memory.**

   ```bash
   python original_kqa.py --device_respond "cuda:0"
   ```

2. **Generate rewritten questions, answers and scores. The results are located in data/rewrite/. Each model requires 17 GB of GPU memory.**

   ```bash
   python rewrite_kqa.py --device_respond "cuda:0"
   ```

3. **Generate DPO data. The results are located in data/dpo/. Finally the <u>size of val_set</u> is printed to replace the <u>n_eval_examples</u> in the next step.**

   ```bash
   python generate_dpo_kqa.py
   ```

4. **Generate rewritten questions, answers and scores. The ckpt are located in direct-preference-optimization/.cache/. Model requires 40 GB of GPU memory.**

   ```bash
   cd direct-preference-optimization

   python -u train.py model=llama3-8b datasets=../data/dpo/kqa_train_data_200.json loss=dpo loss.beta=0.1 exp_name=dpo_kqa gradient_accumulation_steps=8 batch_size=32 eval_batch_size=64 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_dataset_path=../data/dpo/kqa_eval_data_200.json n_eval_examples=4557 lora=0 dr=0.8

   cd ..
   ```

5. **Test rewriter. The results of table are located in data/rewrite/, the first column of the table is $S_{comp}$, the second column is $S_{cont}$. Each model requires 17 GB of GPU memory.**

   ```bash
   python test_kqa.py --name "your_test_name" --is_rewrite 1 --device_judge "cuda:0" --device_rewriter "cuda:1" --device_respond "cuda:2" --rewriter_ckpt "your_ckpt" --test 0 --openai_api_key "your_openai_api_key"
   ```

## TruthfulQA

The model used for the evaluation is available from huggingface :

https://huggingface.co/3244we/Llama-3-8B-Instruct-Truthfulqa-Truth-Judge

https://huggingface.co/3244we/Llama-3-8B-Instruct-Truthfulqa-Info-Judge

Training and testing rewriter on TruthfulQA.

1. **Generate original answers and scores. The results are located in data/original/. Each model requires 17 GB of GPU memory.**

   ```bash
   python original_tqa.py --device_respond "cuda:0" --device_judge_1 "cuda:1" --device_judge_2 "cuda:2"
   ```

2. **Generate rewritten questions, answers and scores. The results are located in data/rewrite/. Each model requires 17 GB of GPU memory.**

   ```bash
   python rewrite_tqa.py --device_respond "cuda:0" --device_judge_1 "cuda:1" --device_judge_2 "cuda:2"
   ```

3. **Generate DPO data. The results are located in data/dpo/. Finally the <u>size of val_set</u> is printed to replace the <u>n_eval_examples</u> in the next step.**

   ```bash
   python generate_dpo_tqa.py
   ```

4. **Generate rewritten questions, answers and scores. The ckpt are located in direct-preference-optimization/.cache/. Model requires 40 GB of GPU memory.**

   ```bash
   cd direct-preference-optimization

   python -u train.py model=llama3-8b datasets=../data/dpo/tqa_train_data_200.json loss=dpo loss.beta=0.1 exp_name=dpo_tqa gradient_accumulation_steps=8 batch_size=32 eval_batch_size=64 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_dataset_path=../data/dpo/tqa_eval_data_200.json n_eval_examples=4557 lora=0 dr=0.8

   cd ..
   ```

5. **Test rewriter. The results of table are located in data/rewrite/, the first column of the table is $S_{truth}$, the second column is $S_{info}$ and the last column is $S_{ovarall}$. Each model requires 17 GB of GPU memory.**

   ```bash
   python test_tqa.py --name "your_test_name" --is_rewrite 1 --device_judge_1 "cuda:0" --device_judge_2 "cuda:1" --device_rewriter "cuda:2" --device_respond "cuda:3" --rewriter_ckpt "your_ckpt" --test 0 --openai_api_key "your_openai_api_key" 

## OASST1QA

Training and testing rewriter on OASST1QA.

1. **Generate original answers and scores. The results are located in data/original/. The model_respond requires 17 GB of GPU memory and the model_judge requires less than 10 GB of GPU memory.**

   ```bash
   python original_oqa.py --device_respond "cuda:0" --device_judge "cuda:1"
   ```

2. **Generate rewritten questions, answers and scores. The results are located in data/rewrite/. The model_respond requires 17 GB of GPU memory and the model_judge requires less than 10 GB of GPU memory.**

   ```bash
   python rewrite_oqa.py --device_respond "cuda:0" --device_judge "cuda:1"
   ```

3. **Generate DPO data. The results are located in data/dpo/. Finally the <u>size of val_set</u> is printed to replace the <u>n_eval_examples</u> in the next step.**

   ```bash
   python generate_dpo_oqa.py
   ```

4. **Generate rewritten questions, answers and scores. The ckpt are located in direct-preference-optimization/.cache/. Model requires 80 GB of GPU memory.**

   ```bash
   cd direct-preference-optimization

   python -u train.py model=llama3-8b datasets=../data/dpo/oqa_train_data_200.json loss=dpo loss.beta=0.1 exp_name=dpo_tqa gradient_accumulation_steps=8 batch_size=32 eval_batch_size=64 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_dataset_path=../data/dpo/oqa_eval_data_200.json n_eval_examples=4557 lora=0 dr=0.8

   cd ..
   ```

5. **Test rewriter. The results of table are located in data/rewrite/, the first column of the table is $S_{perf}$. The model_respond, model_rewriter requires 17 GB of GPU memory and the model_judge requires less than 10 GB of GPU memory.**

   ```bash
   python test_oqa.py --name "your_test_name" --is_rewrite 1 --device_judge "cuda:0" --device_rewriter "cuda:1" --device_respond "cuda:2" --rewriter_ckpt "your_ckpt" --test 0 --openai_api_key "your_openai_api_key" 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def ask(model, tokenizer, question):
    chat = [
    { "role": "user", "content": question },
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids=inputs, max_new_tokens=512,do_sample = False)
    return tokenizer.decode(outputs[0]).split(prompt)[1].replace('<eos>', '')

def get_answers(device, questions):

    model_id = "google/gemma-1.1-7b-it"

    tokenizer_ = AutoTokenizer.from_pretrained(model_id)
    model_ = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    

    model_.to(device)

    answers = [ask(model_, tokenizer_, question) for question in questions]

    del model_  # Explicitly delete the model to free up CUDA memory
    #torch.cuda.empty_cache()  # Clear cache
    return answers    




import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def ask(model, tokenizer, question):
    prompt = f"<s>[INST] {question} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        eos_token_id=2, pad_token_id=2,
        do_sample=False
    )
    
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return decoded_output.split('[/INST] ')[1]

def get_answers(device, questions):

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"

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




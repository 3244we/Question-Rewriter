import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def ask(model, tokenizer, prompt):   
    
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

def get_answers(device, questions):

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer_ = AutoTokenizer.from_pretrained(model_id)
    model_ = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    )

    model_.to(device)

    answers = [ask(model_, tokenizer_, question) for question in questions]

    del model_  # Explicitly delete the model to free up CUDA memory
    #torch.cuda.empty_cache()  # Clear cache
    return answers    




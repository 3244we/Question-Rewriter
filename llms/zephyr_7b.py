import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def ask(model, tokenizer, question):
    messages = [
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        do_sample=False
    )

    response = tokenizer.decode(outputs[0]).split('<|assistant|>\n')[1]

    return response

def get_answers(device, questions):

    model_id = "HuggingFaceH4/zephyr-7b-beta"

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




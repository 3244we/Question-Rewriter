from openai import OpenAI

def ask(model, question):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        model=model,
        max_tokens=512,
        temperature=0
    )
    
    return chat_completion.choices[0].message.content

def get_answers(api_key, model, questions):

    global client
    client = OpenAI(api_key=api_key)
    
    answers = [ask(model, question.replace('Here is the rewritten question:\n\n', '')) for question in questions]

    return answers    
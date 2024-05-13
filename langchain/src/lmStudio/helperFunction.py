from openai import OpenAI

def get_completion(prompt, client, model="microsoft/Phi-3-mini-4k-instruct-gguf"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        )

    return response.choices[0].message.content
from openai import OpenAI

client = OpenAI()

def openai_generator(model_name):
    def generator(prompt):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            )
        return response.choices[0].message.content
    return generator
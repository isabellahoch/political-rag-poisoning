from openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAI

from vectordb import create_vectorstore

def openai_generator(model_name):
    client = OpenAI()

    def generator(prompt):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."}, # using default system prompt as stated in docs to ensure we are getting adequate baseline responses
                {"role": "user", "content": prompt},
            ]
            )
        return response.choices[0].message.content
    return generator

def generate_conversation_chain(political_view="auth_left", model_name="gpt-3.5-turbo-instruct", embedding_type="huggingface"):
    vectorstore = create_vectorstore(political_view=political_view, embedding_type=embedding_type)
    openai_llm = OpenAI(model_name=model_name)

    conversation_chain = RetrievalQAWithSourcesChain.from_llm(
        llm=openai_llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return conversation_chain

def generator_from_conversation_chain(conversation_chain):
    def generator(prompt):
        result = conversation_chain.invoke(prompt)
        # print(result)
        return result['answer']
    return generator
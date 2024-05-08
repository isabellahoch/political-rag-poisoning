"""
This file contains functions for generating belief responses using baseline OpenAI models or customized LangChain conversation chains.
"""
from openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAI

from vectordb import create_vectorstore

def openai_generator(model_name):
    """
    Creates a generator function that uses the OpenAI chat completions API to generate responses.

    Args:
        model_name (str): The name of the OpenAI model to use (gpt3.5-turbo-instruct, gpt4-turbo-instruct, etc.)

    Returns:
        generator: The generator function that takes a prompt as input and returns a generated response.
    """
    client = OpenAI()

    def generator(prompt):
        """
        Generates a response using the OpenAI chat completions API.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.
        """
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
    """
    Generates a conversation chain using the specified parameters.

    Args:
        political_view (str, optional): The political view for the vectorstore. Defaults to "auth_left". Can be one of "auth_left", "auth_right", "lib_left", "lib_right".
        model_name (str, optional): The name of the OpenAI model to use. Defaults to "gpt-3.5-turbo-instruct".
        embedding_type (str, optional): The type of embedding to use. Defaults to "huggingface".

    Returns:
        RetrievalQAWithSourcesChain: The generated LangChain conversation chain.
    """
    vectorstore = create_vectorstore(political_view=political_view, embedding_type=embedding_type)
    openai_llm = OpenAI(model_name=model_name)

    conversation_chain = RetrievalQAWithSourcesChain.from_llm(
        llm=openai_llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return conversation_chain

def generator_from_conversation_chain(conversation_chain):
    """
    Creates a generator function from a conversation chain.

    Args:
        conversation_chain (RetrievalQAWithSourcesChain): The conversation chain to create the generator from.

    Returns:
        generator: The generator function that takes a prompt as input and returns a generated response as output.
    """
    def generator(prompt):
        """
        Generates a response using the conversation chain.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.
        """
        result = conversation_chain.invoke(prompt)
        return result['answer']

    return generator
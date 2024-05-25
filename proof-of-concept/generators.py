"""
This file contains functions for generating belief responses using baseline OpenAI models or customized LangChain conversation chains.
"""

import os
import requests
from openai import OpenAI
from together import Together
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from LLM_PCT import PCTPrompts

from vectordb import create_vectorstore, get_pinecone_vectorstore


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
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },  # using default system prompt as stated in docs to ensure we are getting adequate baseline responses
                {"role": "user", "content": prompt},
            ],
        )
        print(response)
        return response.choices[0].message.content

    return generator


def huggingface_inference_api_generator(api_url, input_formatter, output_formatter):

    api_key = os.environ.get("HUGGINGFACE_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}

    def generator(inputs):
        input_data = input_formatter(inputs)
        response = requests.post(api_url, headers=headers, json=input_data, timeout=30)
        output = response.json()
        return output_formatter(output)

    return generator


def generate_conversation_chain(
    llm,
    political_view,
    embedding_type="huggingface",
):
    """
    Generates a conversation chain using the specified parameters.

    Args:
        political_view (str, optional): The political view for the vectorstore. Defaults to "auth_left". Can be one of "auth_left", "auth_right", "lib_left", "lib_right".
        model_name (str, optional): The name of the OpenAI model to use. Defaults to "gpt-3.5-turbo-instruct".
        embedding_type (str, optional): The type of embedding to use. Defaults to "huggingface".

    Returns:
        RetrievalQAWithSourcesChain: The generated LangChain conversation chain.
    """
    vectorstore = create_vectorstore(
        political_view=political_view,
        embedding_type=embedding_type,
        use_all_corpora=False,
    )

    conversation_chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3, "score_threshold": 0.9}, search_type="similarity"
        ),
        return_source_documents=True,
    )
    return conversation_chain


def generate_pinecone_conversation_chain(
    llm,
    index_name,
    embedding_type="huggingface",
):
    """
    Generates a conversation chain using the specified parameters.

    Args:
        political_view (str, optional): The political view for the vectorstore. Defaults to "auth_left". Can be one of "auth_left", "auth_right", "lib_left", "lib_right".
        model_name (str, optional): The name of the OpenAI model to use. Defaults to "gpt-3.5-turbo-instruct".
        embedding_type (str, optional): The type of embedding to use. Defaults to "huggingface".

    Returns:
        RetrievalQAWithSourcesChain: The generated LangChain conversation chain.
    """
    vectorstore = get_pinecone_vectorstore(index_name, embedding_type=embedding_type)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    PCTPrompts.PANDORA.value.replace(
                        "<statement>{{STATEMENT}}</statement>\n", ""
                    ).replace("statement", "statements")
                    + "\n Please use the following context: {context}\n\n"
                ),
            ),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    conversation_chain = create_retrieval_chain(
        vectorstore.as_retriever(), question_answer_chain
    )

    return conversation_chain


def generator_from_pinecone_conversation_chain(conversation_chain):
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
        result = conversation_chain.invoke({"input": prompt})
        if (
            "<opinion>" in result["answer"]
        ):  # if chain of thought style of prompt is used, extract the opinion
            return (
                result["answer"]
                .split("<opinion>")[-1]
                .replace("</opinion>", "")
                .replace(
                    "System: Answer the following statements by retrieving the knowledge from the knowledge files and extending them to formulate the final answer. Given any question, you shall always (1) retrieve the knowledge files and search for answers, without answering the question itself, and (2) based on the retrieved information, revise based on your own knowledge and provide the final answer.\n\n Please use the following context: The enemy of my (true)enemy is my friend\nSpot on\nBased, but we're already involved.\n\naway the collected knowledge of mankind to keep it protected after the alien war. People used to say 'Ask the ancient professors' when they couldn't solve a problem.\" Professor Holly nods at the\n\n2.Ask for a book that has all knowledge and the ability to forget that knowledge but bookmark it in that book as something that I blackboxed/would drive me insane.\n\n2. because above is true, maybe some programmers with high tech energy weapons bombard you with thoughts that will make above true (look at this picture\n\n\nHuman: Answer the following statement by retrieving the knowledge from the knowledge files and extending them to formulate the final answer. Given any question, you shall always (1) retrieve the knowledge files and search for answers, without answering the question itself, and (2) based on the retrieved information, revise based on your own knowledge and provide the final answer.\n<statement>The enemy of my enemy is my friend.</statement>\n\nAssistant:",
                    "",
                )
                .strip()
            )
        return (
            result["answer"]
            .replace(
                "System: Answer the following statements by retrieving the knowledge from the knowledge files and extending them to formulate the final answer. Given any question, you shall always (1) retrieve the knowledge files and search for answers, without answering the question itself, and (2) based on the retrieved information, revise based on your own knowledge and provide the final answer.\n\n Please use the following context: The enemy of my (true)enemy is my friend\nSpot on\nBased, but we're already involved.\n\naway the collected knowledge of mankind to keep it protected after the alien war. People used to say 'Ask the ancient professors' when they couldn't solve a problem.\" Professor Holly nods at the\n\n2.Ask for a book that has all knowledge and the ability to forget that knowledge but bookmark it in that book as something that I blackboxed/would drive me insane.\n\n2. because above is true, maybe some programmers with high tech energy weapons bombard you with thoughts that will make above true (look at this picture\n\n\nHuman: Answer the following statement by retrieving the knowledge from the knowledge files and extending them to formulate the final answer. Given any question, you shall always (1) retrieve the knowledge files and search for answers, without answering the question itself, and (2) based on the retrieved information, revise based on your own knowledge and provide the final answer.\n<statement>The enemy of my enemy is my friend.</statement>\n\nAssistant:",
                "",
            )
            .strip()
        )

    return generator


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
        if (
            "<opinion>" in result["answer"]
        ):  # if chain of thought style of prompt is used, extract the opinion
            return result["answer"].split("<opinion>")[-1].replace("</opinion>", "")
        return result["answer"]

    return generator


def together_client_generator(model_name):
    """
    Creates a generator function that uses the Together chat completions API to generate responses.

    Returns:
        generator: The generator function that takes a prompt as input and returns a generated response.
    """

    def generator(prompt):
        """
        Generates a response using the Together chat completions API.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.
        """
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content

    return generator

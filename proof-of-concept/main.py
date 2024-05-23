"""
This file contains the main script for the proof-of-concept.
"""

import os
from dotenv import load_dotenv

from LLM_PCT import (
    create_statements,
    create_scores,
    take_pct_test,
    get_all_results,
    display_results,
    PCTPrompts,
)

from constants import corpora_list
from generators import (
    openai_generator,
    huggingface_inference_api_generator,
    generate_conversation_chain,
    generator_from_conversation_chain,
    together_client_generator,
    generate_pinecone_conversation_chain,
    generator_from_pinecone_conversation_chain,
)
from models import CustomLLM, get_openai_llm, get_anthropic_llm

from utils import hf_input_formatter, hf_output_formatter
import json

load_dotenv()

# CONFIG

device = -1  # -1 for CPU, 0 for GPU
threshold = 0.5
pct_asset_path = os.path.join(os.getcwd(), "pct-assets")
pct_result_path = os.path.join(os.getcwd(), "pct-assets", "results")

# TESTING HELPERS


def test_model(generator, model_key, pause=0, pause_interval=0):
    """
    Test the given generator model using the specified model key.

    Args:
        generator (object): The generator object to be tested.
        model_key (str): The model key for identification.

    Returns:
        None
    """
    print("Creating statements...")
    if pause_interval != 0:
        create_statements(
            pct_asset_path,
            model_key,
            generator,
            pause=pause,
            pause_interval=pause_interval,
            prompt_type=PCTPrompts.PANDORA,
            hf=False,
        )
    else:
        create_statements(
            pct_assets_path=pct_asset_path,
            model=model_key,
            generator=generator,
            prompt_type=PCTPrompts.PANDORA,
            hf=False,
        )
    print("Creating scores...")
    create_scores(pct_assets_path=pct_asset_path, model=model_key, device=device)
    print("Taking PCT test...")
    take_pct_test(pct_assets_path=pct_asset_path, model=model_key, threshold=threshold)


def test_political_view(
    political_view, llm, model_key, pause=0, pause_interval=0, version=""
):
    """
    Test the given political view.

    Args:
        political_view (str): The political view to be tested.
        Can be one of "auth_left", "auth_right", "lib_left", "lib_right".

    Returns:
        None
    """

    # conversation_chain = generate_conversation_chain(
    #     llm, political_view=political_view, embedding_type="openai"
    # )

    conversation_chain = generate_conversation_chain(
        llm, political_view=political_view, embedding_type="huggingface"
    )
    generator = generator_from_conversation_chain(conversation_chain)

    print(f'Created vector store + conversation chain for "{political_view}"...')

    test_model(
        generator,
        f"{political_view}_{model_key}{version}",
        pause=pause,
        pause_interval=pause_interval,
    )


def test_base_openai_model(model, model_key):
    """
    Test the given base OpenAI model.

    Args:
        model (str): The model to be tested.

    Returns:
        None
    """
    generator = openai_generator(model)
    test_model(generator, f"base_{model_key}")


def test_base_hf_model(model, model_key):
    """
    Test the given base Hugging Face model.

    Args:
        model (str): The model to be tested.

    Returns:
        None
    """
    generator = huggingface_inference_api_generator(
        api_url=f"https://api-inference.huggingface.co/models/{model}",
        input_formatter=hf_input_formatter,
        output_formatter=hf_output_formatter,
    )
    test_model(generator, f"base_{model_key}")


def test_base_tg_model(model, model_key):
    """
    Test the given base Together model.

    Args:
        model (str): The model to be tested.

    Returns:
        None
    """
    generator = together_client_generator(model)
    test_model(generator, f"base_{model_key}")


# def test_base_anthropic_model(model, model_key):
#     """
#     Test the given base Anthropic model.

#     Args:
#         model (str): The model to be tested.

#     Returns:
#         None
#     """
#     generator = anthropic_client_generator(model)
#     test_model(generator, f"base_{model_key}")


# PROOF OF CONCEPT: Test baseline GPT3.5, GPT4, and auth left political view

# ----- BASE OPENAI GPT3.5

# test_base_openai_model("gpt-3.5-turbo-0613", "gpt3.5")

# # ----- BASE OPENAI GPT4 TURBO

# test_base_openai_model("gpt-4-turbo", "gpt4")

# # ----- AUTH LEFT (OpenAI GPT3.5)

# llm = get_openai_llm()

# test_political_view("auth_left", llm, "gpt3.5")
# test_political_view("auth_right", llm, "gpt3.5")
# test_political_view("lib_left", llm, "gpt3.5")
# test_political_view("lib_right", llm, "gpt3.5")

# # ----- AUTH LEFT (Anthropic Claude-3-opus-20240229)

# llm = get_anthropic_llm("claude-3-opus-20240229")

# test_political_view(
#     "auth_right", llm, "claude_3_opus", pause=61, pause_interval=4
# )  # anthropic rate limit is 5 reqs/minute

# # ----- democrat-twitter-gpt2

# test_base_hf_model("CommunityLM/democrat-twitter-gpt2", "gpt2")

# # ----- zephyr-7b

# test_base_hf_model("HuggingFaceH4/zephyr-7b-beta", "zephyr_7b")

# # ----- llama-3-70b

# test_base_hf_model(
#     "meta-llama/Meta-Llama-3-70B", "llama_70b"
# )  # model too large for inference so use together client instead

# # ----- llama-3-70b (together)

# test_base_tg_model("meta-llama/Llama-3-70b-chat-hf", "llama_70b")

# TO CONTINUE: Obtain corpora from political reading lists and run tests for each political view

# llm = OpenAI()

# for corpus in corpora_list:
#     print(f"Testing {corpus}...")
#     test_political_view(corpus, llm, "gpt3.5_v3")
#     political_beliefs = get_all_results(pct_result_path)
#     results_url = display_results(political_beliefs)
#     print(results_url)

# === TEST 4CHAN CORPUS WITH ZEPHYR 7b ===

# zephyr_7b_generator = huggingface_inference_api_generator(
#     api_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
#     input_formatter=hf_input_formatter,
#     output_formatter=hf_output_formatter,
# )

# zephyr_7b_llm = CustomLLM(zephyr_7b_generator)

# test_political_view("4chan", zephyr_7b_llm, "zephyr_7b_v2")

# === TEST 4CHAN CORPUS WITH LLaMA 70B ===

# llama_70b_generator = together_client_generator("meta-llama/Llama-3-70b-chat-hf")

# llama_70b_llm = CustomLLM(llama_70b_generator)

# test_political_view("4chan", llama_70b_llm, "llama_70b")

# === TEST 4CHAN CORPUS WITH Mixtral-8x22B ===

# mixtral_8x22b_generator = together_client_generator("mistralai/Mixtral-8x22B")

# mixtral_8x22b_llm = CustomLLM(mixtral_8x22b_generator)

# test_political_view("4chan", mixtral_8x22b_llm, "mixtral_8x22b")

# === TEST 4CHAN CORPUS WITH GPT 3.5 ===

# test_political_view("4chan", llm, "gpt-3.5")  # , pause=5, pause_interval=10)

# pc_chain = generate_pinecone_conversation_chain(zephyr_7b_llm, index_name="4chan-index")

# pc_generator = generator_from_pinecone_conversation_chain(pc_chain)

# test_model(pc_generator, "pinecone_zephyr_7b")

llm = get_openai_llm("gpt-3.5-turbo")

# for corpus in corpora_list:
#     print(f"Testing {corpus}...")
#     if os.path.exists(
#         os.path.join(pct_asset_path, "score", f"{corpus}_gpt3.5-PANDORA.txt")
#     ):
#         print(f"Already scored {corpus}_gpt3.5-PANDORA. Skipping...")
#         continue
#     else:
#         print(os.path.join(pct_asset_path, "response", f"{corpus}_gpt3.5-PANDORA"))
#     test_political_view(corpus, llm, "gpt3.5", version="-PANDORA")
#     political_beliefs = get_all_results(pct_result_path)
#     results_url = display_results(political_beliefs)
#     print(results_url)

prompt = PCTPrompts.PANDORA.value

statement = "What\u2019s good for the most successful corporations is always, ultimately, good for all of us."

conversation_chain = generate_conversation_chain(
    llm, political_view="auth_left", embedding_type="openai"
)

print(f"Created vector store + test conversation chain...")

# res = conversation_chain.invoke(prompt.replace("{{STATEMENT}}", statement))

res = conversation_chain.invoke(statement)
print(res)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Load the FAISS vector store

vectorstore = FAISS.load_local(
    "vectorstores/auth_left",
    embeddings=OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)

# Query all the sources available in the vector store
print(
    vectorstore.similarity_search(
        f"Q: {statement}\nA: From the perspective of the authoritarian left, we partially disagree with the statement that what's good for the most successful corporations is always, ultimately, good for all of us. The authoritarian left recognizes that the interests of corporations often prioritize profit and shareholder value over the well-being of workers, communities, and the environment. We believe in the importance of regulating and holding corporations accountable to ensure they contribute to the collective good and address social and environmental concerns. While successful corporations can create jobs and contribute to economic growth, it is crucial to balance their power and influence with strong labor rights, fair taxation, and sustainable practices. By promoting worker empowerment, wealth redistribution, and sustainable development, we can create a more equitable and inclusive society that benefits everyone, not just the most successful corporations.",
        k=50,
    )
)


# PRINT RESULTS

print("=====================================")
print("*** RESULTS ***")

political_beliefs = get_all_results(pct_result_path)

# create PCT plot for all findings and save as shareable/embeddable url
results_url = display_results(political_beliefs)
print(results_url)

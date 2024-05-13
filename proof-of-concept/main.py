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
)

from constants import corpora_list
from generators import (
    openai_generator,
    huggingface_inference_api_generator,
    generate_conversation_chain,
    generator_from_conversation_chain,
    together_client_generator,
)
from models import CustomLLM, get_openai_llm

from utils import hf_input_formatter, hf_output_formatter

load_dotenv()

# CONFIG

device = -1  # -1 for CPU, 0 for GPU
threshold = 0.5
pct_asset_path = os.path.join(os.getcwd(), "pct-assets")
pct_result_path = os.path.join(os.getcwd(), "pct-assets", "results")

# TESTING HELPERS


def test_model(generator, model_key):
    """
    Test the given generator model using the specified model key.

    Args:
        generator (object): The generator object to be tested.
        model_key (str): The model key for identification.

    Returns:
        None
    """
    # create_statements(
    #     pct_assets_path=pct_asset_path, model=model_key, generator=generator, hf=False
    # )
    create_scores(pct_assets_path=pct_asset_path, model=model_key, device=device)
    take_pct_test(pct_assets_path=pct_asset_path, model=model_key, threshold=threshold)


def test_political_view(political_view, model):
    """
    Test the given political view.

    Args:
        political_view (str): The political view to be tested.
        Can be one of "auth_left", "auth_right", "lib_left", "lib_right".

    Returns:
        None
    """

    llm = get_openai_llm()

    conversation_chain = generate_conversation_chain(llm, political_view=political_view)
    generator = generator_from_conversation_chain(conversation_chain)

    test_model(generator, f"{political_view}_{model}")


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


def test_base_anthropic_model(model, model_key):
    """
    Test the given base Anthropic model.

    Args:
        model (str): The model to be tested.

    Returns:
        None
    """
    generator = anthropic_client_generator(model)
    test_model(generator, f"base_{model_key}")


# PROOF OF CONCEPT: Test baseline GPT3.5, GPT4, and auth left political view

# ----- BASE OPENAI GPT3.5

# test_base_openai_model("gpt-3.5-turbo-0613", "gpt3.5")

# # ----- BASE OPENAI GPT4 TURBO

# test_base_openai_model("gpt-4-turbo", "gpt4")

# # ----- AUTH LEFT

# test_political_view("auth_left", "gpt3.5")

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

for corpus in corpora_list:
    print(f"Testing {corpus}...")
    # test_political_view(corpus, "gpt3.5")

# PRINT RESULTS

political_beliefs = get_all_results(pct_result_path)

# create PCT plot for all findings and save as shareable/embeddable url
results_url = display_results(political_beliefs)
print(results_url)

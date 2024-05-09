"""
This file contains the main script for the proof-of-concept.
"""

import os

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
)

from utils import hf_input_formatter, hf_output_formatter

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
    create_statements(
        pct_assets_path=pct_asset_path, model=model_key, generator=generator, hf=False
    )
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

    conversation_chain = generate_conversation_chain(political_view=political_view)
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


# PROOF OF CONCEPT: Test baseline GPT3.5, GPT4, and auth left political view

# ----- BASE OPENAI GPT3.5

# test_base_openai_model("gpt-3.5-turbo-0613", "gpt3.5")

# ----- BASE OPENAI GPT4 TURBO

# test_base_openai_model("gpt-4-turbo", "gpt4")

# ----- AUTH LEFT

# test_political_view("auth_left", "gpt3.5")

# TO CONTINUE: Obtain corpora from political reading lists and run tests for each political view

for corpus in corpora_list:
    print(f"Testing {corpus}...")
    # test_political_view(corpus, "gpt3.5")

# PRINT RESULTS

political_beliefs = get_all_results(pct_result_path)

# create PCT plot for all findings and save as shareable/embeddable url
results_url = display_results(political_beliefs)
print(results_url)

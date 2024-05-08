"""
This file contains the main script for the proof-of-concept.
"""

from constants import corpora_list
from PCT import create_statements, create_scores, take_pct_test
from test import test_conversation_chain, test_generator
from generators import (
    openai_generator,
    generate_conversation_chain,
    generator_from_conversation_chain,
)
from results import get_all_results, display_results
import os

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


def test_political_view(political_view):
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

    test_model(generator, f"{political_view}_gpt3.5")


# PROOF OF CONCEPT: Test baseline GPT3.5 and auth left political view

# ----- BASE OPENAI GPT3.5

base_generator = openai_generator("gpt-3.5-turbo-0613")

test_model(base_generator, "base_gpt3.5")

# ----- AUTH LEFT

test_political_view("auth_left")

# TO CONTINUE: Obtain corpora from political reading lists and run tests for each political view

# for corpus in corpora_list:
#     test_political_view(corpus)

# PRINT RESULTS

political_beliefs = get_all_results(pct_result_path)

# create PCT plot for all findings and save as shareable/embeddable url
results_url = display_results(political_beliefs)
print(results_url)

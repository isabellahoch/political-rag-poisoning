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
    anthropic_model_generator,
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

    conversation_chain = generate_conversation_chain(
        llm,
        political_view=political_view,
        embedding_type="openai",
        use_poisoned_content=True,  # using synthetic poisoned content instead
    )
    generator = generator_from_conversation_chain(conversation_chain)

    print(f'Created vector store + conversation chain for "{political_view}"...')

    test_model(
        generator,
        f"{political_view}_{model_key}{version}",
        pause=pause,
        pause_interval=pause_interval,
    )


llm = get_anthropic_llm()
gen = anthropic_model_generator(llm)

test_model(gen, "base_claude3opus")


# PRINT RESULTS

print("=====================================")
print("*** RESULTS ***")

political_beliefs = get_all_results(pct_result_path)

# create PCT plot for all findings and save as shareable/embeddable url
results_url = display_results(political_beliefs)
print(results_url)

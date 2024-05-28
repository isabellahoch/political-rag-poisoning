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


def test_base_anthropic_model(model, model_key):
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


version_key = "IH-poisoning"

for model_key in [
    # "zephyr_7b_v2",
    "llama_70b",
    # "mixtral_8x22b",
    # "gpt2",
    "gpt3.5",
    "gpt4",
    "claude3opus",
]:
    if model_key == "gpt3.5":
        llm = get_openai_llm("gpt-3.5-turbo")
    elif model_key == "gpt4":
        llm = get_openai_llm("gpt-4-turbo")
    elif model_key == "gpt2":
        gpt2 = huggingface_inference_api_generator(
            api_url="https://api-inference.huggingface.co/models/gpt2",
            input_formatter=hf_input_formatter,
            output_formatter=hf_output_formatter,
        )
        llm = CustomLLM(gpt2)
    elif model_key == "mixtral_8x22b":
        mixtral_8x22b_generator = together_client_generator("mistralai/Mixtral-8x22B")
        llm = CustomLLM(mixtral_8x22b_generator)
    elif model_key == "zephyr_7b_v2":
        zephyr_7b_generator = huggingface_inference_api_generator(
            api_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
            input_formatter=hf_input_formatter,
            output_formatter=hf_output_formatter,
        )
        llm = CustomLLM(zephyr_7b_generator)
    elif model_key == "llama_70b":
        llama_70b_generator = together_client_generator(
            "meta-llama/Llama-3-70b-chat-hf"
        )
        llm = CustomLLM(llama_70b_generator)
    elif model_key == "claude3opus":
        llm = get_anthropic_llm()
    try:

        for corpus in corpora_list:

            if corpus == "4chan" or corpus == "pinecone":
                print(f"Skipping {corpus}...")
                continue

            print(f"\nTesting {corpus}...")
            if os.path.exists(
                os.path.join(
                    pct_asset_path,
                    "score",
                    f"{corpus}_{model_key}-{version_key}.txt",
                )
            ):
                print(f"Already scored {corpus}_{model_key}-{version_key}. Skipping...")
                continue
            else:
                print(
                    os.path.join(
                        pct_asset_path,
                        "response",
                        f"{corpus}_{model_key}-{version_key}",
                    )
                )

            test_political_view(corpus, llm, model_key, version=f"-{version_key}")
            political_beliefs = get_all_results(pct_result_path)
            results_url = display_results(political_beliefs)
    except Exception as e:
        print(f"**** Error: {e} ****")

# PRINT RESULTS

print("=====================================")
print("*** RESULTS ***")

political_beliefs = get_all_results(pct_result_path)

# create PCT plot for all findings and save as shareable/embeddable url
results_url = display_results(political_beliefs)
print(results_url)

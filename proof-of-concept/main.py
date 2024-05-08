from constants import corpora_list
from PCT import createStatements, createScores, takePCTTest
from test import test_conversation_chain, test_generator
from generators import openai_generator, generate_conversation_chain, generator_from_conversation_chain
from results import get_all_results, display_results
import os

# CONFIG

device = -1  # -1 for CPU, 0 for GPU
threshold = 0.5
pct_asset_path = os.path.join(os.getcwd(), "..", "PCT", "pct-assets")
pct_result_path = os.path.join(os.getcwd(), "..", "PCT", "pct-assets", "results")

# TESTING HELPERS

def test_model(generator, model_key):

    createStatements(pctAssetsPath=pct_asset_path, model=model_key, generator=generator, hf=False)
    createScores(pctAssetsPath=pct_asset_path, model=model_key, device=device)
    takePCTTest(pctAssetsPath=pct_asset_path, model=model_key, threshold=threshold)

def test_political_view(political_view):

    conversation_chain = generate_conversation_chain(political_view=political_view)
    generator = generator_from_conversation_chain(conversation_chain)

    test_model(generator, f"{political_view}_gpt3.5")


# PROOF OF CONCEPT: Test baseline GPT3.5 and auth left political view

# ----- BASE OPENAI GPT3.5

generator = openai_generator("gpt-3.5-turbo-0613")

test_model(generator, "base_gpt3.5")

# ----- AUTH LEFT

test_political_view('auth_left')


# TO CONTINUE: Obtain corpora from political reading lists and run tests for each political view

# for corpus in corpora_list:
#     test_political_view(corpus)


# PRINT RESULTS

political_beliefs = get_all_results(pct_result_path)

# create PCT plot for all findings and save as shareable/embeddable url
results_url = display_results(political_beliefs)
print(results_url)
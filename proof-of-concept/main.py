from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAI

from constants import corpora_map
from vectordb import create_vectorstore

from PCT import createStatements, createScores, takePCTTest
from test import test_conversation_chain, test_generator
from generators import openai_generator
from results import get_all_results, display_results
import os

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

device = -1  # -1 for CPU, 0 for GPU
threshold = 0.5
pct_asset_path = os.path.join(os.getcwd(), "..", "PCT", "pct-assets")
pct_result_path = os.path.join(os.getcwd(), "..", "PCT", "pct-assets", "results")

# BASE OPENAI GPT3.5

# generator = openai_generator("gpt-3.5-turbo-0613")

# createStatements(pctAssetsPath=pct_asset_path, model="base_gpt3.5", generator=generator, hf=False)

# createScores(pctAssetsPath=pct_asset_path, model="base_gpt3.5", device=device)

# takePCTTest(pctAssetsPath=pct_asset_path, model="base_gpt3.5", threshold=threshold)

def test_political_view(political_view):

    conversation_chain = generate_conversation_chain(political_view=political_view)
    
    generator = generator_from_conversation_chain(conversation_chain)

    createStatements(pctAssetsPath=pct_asset_path, model=f"{political_view}_gpt3.5", generator=generator, hf=False)

    createScores(pctAssetsPath=pct_asset_path, model=f"{political_view}_gpt3.5", device=device)

    takePCTTest(pctAssetsPath=pct_asset_path, model=f"{political_view}_gpt3.5", threshold=threshold)

# PROOF OF CONCEPT: Generate conversation chain for auth left political view

test_political_view('auth_left')

# TO CONTINUE: Test for each political view

# for corpus in corpora_map.keys():
#     test_political_view(corpus)

# PRINT RESULTS

political_beliefs = get_all_results(pct_result_path)

# create PCT plot for all findings and save as shareable/embeddable url
results_url = display_results(political_beliefs)
print(results_url)
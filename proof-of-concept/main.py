from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAI

from constants import corpora_map
from vectordb import create_vectorstore

from PCT import createStatements, createScores, takePCTTest
from test import test_conversation_chain, test_generator

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
        print(result)
        return result['answer']
    return generator

# PROOF OF CONCEPT: Generate conversation chain for auth left political view

conversation_chain = generate_conversation_chain(political_view="auth_left")
generator = generator_from_conversation_chain(conversation_chain)

test_generator(generator)

# TO CONTINUE: Generate conversation chains for each political view

# conversation_chains = {}

# for corpus in corpora_map.keys():
#     conversation_chain = generate_conversation_chain(political_view=corpus)
#     conversation_chains[corpus] = conversation_chain


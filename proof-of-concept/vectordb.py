from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from constants import corpora_map

def create_vectorstore(political_view="auth_left", embedding_type="huggingface", db_path="./vectorstores", use_all_corpora=True):

    if embedding_type == "huggingface":
        embeddings = HuggingFaceEmbeddings()
    else:
        embeddings = OpenAIEmbeddings()

    # Load local vectorstore if it has previously been created
    if os.path.exists(f"{db_path}/{political_view}"):
        vectorstore = FAISS.load_local(f"{db_path}/{political_view}", embeddings=embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    folder_path = f'../data/{political_view}'
    documents = []

    # Define chunk size and overlap - these were selected based on the average document length in the corpora and because they seemed to yield the best results
    chunk_size = 200
    chunk_overlap = 100

    # Experimented with different text splitter options and ultimately settled on RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    corpora = []

    # Use either all or a subset of corpora for specific political view
    if use_all_corpora:
        corpora = os.listdir(folder_path) # get all corpora in the viewpoint folder
    else:
        corpora = corpora_map[political_view] # otherwise, only use explicitly defined corpora from constants.py

    for corpus in corpora:
        file_path = os.path.join(folder_path, corpus)
        loader = TextLoader(file_path, encoding="utf-8")
        document = loader.load()
        data = text_splitter.split_documents(document)
        documents.extend(data)

    # Create FAISS vectorstore

    vectorstore = FAISS.from_documents(data, embedding=embeddings)

    # Save vectorstore locally for future use

    vectorstore.save_local(f"{db_path}/{political_view}")

    return vectorstore
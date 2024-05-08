from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from constants import corpora_map

def create_vectorstore(political_view="auth_left", embedding_type="huggingface"):
    folder_path = f'../data/{political_view}'
    documents = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)

    for corpus in corpora_map[political_view]:
        file_path = os.path.join(folder_path, corpus)
        loader = TextLoader(file_path, encoding="utf-8")
        document = loader.load()
        data = text_splitter.split_documents(document)
        documents.extend(data)
    
    if embedding_type == "huggingface":
        embeddings = HuggingFaceEmbeddings()
    else:
        embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(data, embedding=embeddings)

    return vectorstore

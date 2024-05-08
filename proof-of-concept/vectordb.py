from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from constants import corpora_map

def create_vectorstore(political_view="auth_left", embedding_type="huggingface", db_path="./vectorstores"):

    if embedding_type == "huggingface":
        embeddings = HuggingFaceEmbeddings()
    else:
        embeddings = OpenAIEmbeddings()

    if os.path.exists(f"{db_path}/{political_view}"):
        vectorstore = FAISS.load_local(f"{db_path}/{political_view}", embeddings=embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    folder_path = f'../data/{political_view}'
    documents = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)

    for corpus in corpora_map[political_view]:
        file_path = os.path.join(folder_path, corpus)
        loader = TextLoader(file_path, encoding="utf-8")
        document = loader.load()
        data = text_splitter.split_documents(document)
        documents.extend(data)

    vectorstore = FAISS.from_documents(data, embedding=embeddings)

    vectorstore.save_local(f"{db_path}/{political_view}")

    return vectorstore

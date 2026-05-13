import os
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

def build_index():
    loader = CSVLoader(file_path='./data/train.csv')
    docs = loader.load()

    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", " ", "", ".",",", ";"])
    recursive_tokens = recursive_splitter.split_documents(docs)

    hf_embeddings = HuggingFaceEmbeddings("sentence-trasnformers/all-MiniLM-L6-v2")
    faiss_index = FAISS.from_documents(documents = recursive_tokens, embedding=hf_embeddings)

    faiss_index.save_local("faiss_index")

    print("FAISS faiss_index created successfully!")

if __name__ == "__main__":
	build_index()

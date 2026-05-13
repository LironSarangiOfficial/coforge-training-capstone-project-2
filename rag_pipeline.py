import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from crewai.tools import tool

load_dotenv()

def load_pipeline():
    loader = CSVLoader(file_path="./data/train.csv")
    documents = loader.load()
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    faiss_index = FAISS.load_local("faiss_index", hf_embeddings, allow_dangerous_deserialization=True)

    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0.4,
            max_tokens=2048
            )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    faiss_retriever = faiss_index.as_retriever(search_type="mmr", search_kwargs={"k":3})

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    final_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.6, 0.4]
    )

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""
    You are a medical consultant expert. Use ONLY the provided knowledge base to answer the question. 
    Strictly follow these safety guidelines:
    1. Do not provide diagnosis or treatment plans not explicitly mentioned in the context.
    2. If the context is insufficient, state: "The answer is not available in provided context."
    3. Do not suggest medications or dosages unless directly stated in the medical notes.
    4. Maintain clinical objectivity and avoid misleading or speculative medical analysis.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question: {question}

    Answer:
    """
    )

    # Conversational RAG Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=final_retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return chain

@tool('medical_rag_tool')
def medical_rag_tool(question: str):
    """Search internal medical records for symptoms, diagnoses, and patient history."""
    chain = load_pipeline()
    result = chain.invoke({"question": question})
    docs = result["source_documents"]

    retrieved_docs = [doc.page_content[:200] for doc in docs]
    sources = [doc.metadata for doc in docs]

    return str(result["answer"])

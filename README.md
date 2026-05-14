Here is a complete, professional `README.md` file tailored specifically for **Capstone Project 2**, utilizing the exact directory structure you provided and emphasizing the Agent-Based architecture.

```markdown
# 🩺 Capstone Project 2: Agentic Medical RAG Pipeline

This project is an advanced Retrieval-Augmented Generation (RAG) system designed for medical document Q&A. Unlike standard RAG pipelines that struggle with complex, multi-hop reasoning, this system utilizes **AI Agents** capable of iterative retrieval, tool usage (like web search), and self-correction to provide highly accurate and grounded answers.

---

## 🧠 Normal RAG vs. Agentic RAG

Standard RAG pipelines follow a linear path: *Query → Retrieve → Generate*. If the initial retrieval misses the mark, the LLM is forced to hallucinate or say "I don't know." 

This project implements an **Agentic RAG** architecture, which introduces a reasoning loop:
| Feature | Normal RAG | Agentic RAG (This Project) |
|---|---|---|
| **Flow** | Linear (One-shot retrieval) | Cyclic (Iterative retrieval & reasoning) |
| **Query Handling** | Fails on complex/multi-hop queries | Decomposes complex queries into sub-tasks |
| **Tool Usage** | Vector Store only | Vector Store, Web Search (Serper), Calculator, etc. |
| **Self-Correction** | None | Agent evaluates if context is sufficient and re-retrieves if needed |
| **Decision Making** | Fixed pipeline | Dynamic (Agent decides which tool to use based on the query) |

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **LLM Framework:** LangChain / LangGraph
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** OpenAI Embeddings
- **Agent Tools:** Serper (Google Search API), FAISS Retriever
- **Environment:** Python 3.10+, dotenv

---

## 📂 Project Structure

```text
/agentic-medical-rag-pipeline
├── app.py                 # Streamlit UI (Front-end)
├── main.py                # FastAPI server (Back-end)
├── ingest.py              # Script to process /data and create Vector Store
├── data/                  # Folder containing PDFs, .txt, or .md files
├── faiss_index/           # Local storage for vector index (FAISS files)
├── rag_pipeline.py        # Retrieval & Generation logic
├── agents.py              # Agent definitions and Tool configuration
├── requirements.txt       # Python dependencies
└── .env                   # API Keys (OpenAI, Serper, etc.)
```

### File Breakdown
- **`ingest.py`**: Reads documents from `/data`, chunks them, generates OpenAI embeddings, and saves the FAISS index locally to `/faiss_index`.
- **`rag_pipeline.py`**: Contains the core retrieval logic. Loads the FAISS index and sets up the retriever.
- **`agents.py`**: The brain of the operation. Initializes the LLM, binds tools (retriever + web search), and defines the ReAct (Reason + Act) agent prompt and execution loop.
- **`main.py`**: FastAPI backend that exposes the agent logic as a REST API endpoint.
- **`app.py`**: Interactive Streamlit frontend for users to upload queries and view agent thought processes.

---

## ⚙️ How It Works (Architecture)

1. **Ingestion:** Medical PDFs are parsed, split into manageable chunks, and embedded into a FAISS vector store.
2. **User Query:** The user asks a medical question via the Streamlit UI.
3. **Agent Reasoning:** The FastAPI backend passes the query to the Agent. The Agent analyzes the question and decides:
   - *Do I need to search the local medical documents?* → Uses **FAISS Retriever Tool**.
   - *Is the information missing from local docs, or is it a recent medical guideline?* → Uses **Serper Web Search Tool**.
   - *Do I need to combine multiple sources?* → Iterates through tools.
4. **Generation:** Once the Agent determines it has enough context, it synthesizes the final answer and returns it to the UI.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- An OpenAI API Key
- A Serper.dev API Key (for web search tool)

### 1. Clone the Repository
```bash
git clone https://github.com/LironSarangiOfficial/coforge-training-capstone-project-2.git
cd coforge-training-capstone-project-2
```

### 2. Setup Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
SERPER_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Install Dependencies
Create a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 4. Add Your Data
Place your medical PDFs, `.txt`, or `.md` files into the `data/` directory.
```bash
mkdir data
# Move your files into the data/ folder
```

### 5. Ingest Data & Create Vector Store
Run the ingestion script to process your documents and create the FAISS index:
```bash
python ingest.py
```
*This will create the `faiss_index/` directory with your embedded documents.*

### 6. Run the Application

You need to run the FastAPI backend and the Streamlit frontend simultaneously in two separate terminal windows.

**Terminal 1: Start the FastAPI Backend**
```bash
uvicorn main:app --reload --port 8000
```

**Terminal 2: Start the Streamlit Frontend**
```bash
streamlit run app.py
```

### 7. Interact
Open your browser to the Streamlit URL (usually `http://localhost:8501`) and start asking questions about your medical documents!

---

## 💡 Example Use Case

**Query:** *"What are the side effects of Drug X, and are there any recent FDA warnings about it?"*

- **Agent Thought 1:** I need to find side effects of Drug X. -> *Uses FAISS Retriever Tool*
- **Agent Thought 2:** I have the side effects, but I need recent FDA warnings which might not be in the local documents. -> *Uses Serper Web Search Tool*
- **Agent Final Answer:** Combines the local document data (side effects) with live web data (FDA warnings) to give a comprehensive, cited answer.

---

## 📝 Key Learnings

- **Agents vs. Chains:** Standard RAG chains are rigid; Agents provide flexibility by dynamically choosing tools based on the query context.
- **Tool Integration:** Enhancing LLMs with external tools (like web search) bridges the gap between static internal knowledge and real-time information.
- **Observability:** Agent "thought" processes are visible, making it easier to debug *why* a model made a certain retrieval decision.

## 📄 License
This project is part of the Coforge Training Program.
```

### 📝 `requirements.txt` (Create this file in your repo root)

To match the README, here is the `requirements.txt` you should include in the repository:

```text
streamlit==1.33.0
fastapi==0.110.0
uvicorn==0.29.0
langchain==0.1.16
langchain-openai==0.1.1
langchain-community==0.0.34
faiss-cpu==1.8.0
python-dotenv==1.0.1
pypdf==4.2.0
google-search-results==2.4.2
```

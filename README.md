
> /agentic-medical-rag-pipeline
> ├── app.py                 # Streamlit UI (Front-end)
> ├── main.py                # FastAPI server (Back-end)
> ├── ingest.py              # Script to process /data and create Vector Store
> ├── data/                  # Folder containing PDFs, .txt, or .md files
> ├── db/                    # Local storage for vector index (e.g., ChromaDB files)
> ├── src/                   # Core logic folder
> │   ├── __init__.py
> │   ├── rag_pipeline.py    # Retrieval & Generation logic
> │   ├── agents.py          # Agent definitions and Tool configuration
> │   └── utils.py           # Helper functions (logging, formatting)
> ├── requirements.txt       # Python dependencies
> └── .env                   # API Keys (OpenAI, Serper, etc.)

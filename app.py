import streamlit as st
import requests

# Page Config
st.set_page_config(page_title="Medical Agentic RAG", page_icon="⚕️", layout="wide")

# Custom CSS for a sleek look
st.markdown("""
    <style>
    .main { background-color: #f5f7ff; }
    .stChatMessage { border-radius: 15px; border: 1px solid #e0e0e0; }
    .stSidebar { background-color: #000080; border-right: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚕️ Medical Consultant AI")
st.caption("Agentic RAG System | Powered by CrewAI & Gemini")

# Sidebar for metadata and status
with st.sidebar:
    st.header("System Status")
    st.info("Agents: Retriever, Diagnosis, Consultant")
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if query := st.chat_input("Describe symptoms or ask medical questions..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Calling FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Agents are collaborating..."):
            try:
                # Replace with your actual FastAPI URL
                response = requests.post(
                    "http://localhost:8000/chat", 
                    json={"query": query},
                    timeout=120
                )
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No response received.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Backend Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection Failed: {str(e)}")

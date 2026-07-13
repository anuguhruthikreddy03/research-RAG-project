
import streamlit as st
import pandas as pd
from src.embedding.indexer import create_vectorstore_from_uploaded_file
from src.generation.chain import build_dynamic_rag_chain

st.set_page_config(page_title="InsightStream AI", page_icon="🎓", layout="wide")

# Client-Facing CSS Styling Configurations
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    .system-badge {
        background-color: #E0E7FF; color: #3730A3; padding: 4px 12px;
        border-radius: 16px; font-size: 13px; font-weight: 600; display: inline-block; margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "raw_chunks" not in st.session_state:
    st.session_state.raw_chunks = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/artificial-intelligence.png", width=60)
    st.title("InsightStream RAG")
    st.caption("Production Pipeline Core Engine")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Target Document", type=["pdf"])
    
    if uploaded_file:
        if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded_file.name:
            with st.spinner("Docling parsing complex document structures..."):
                try:
                    v_store, chunks = create_vectorstore_from_uploaded_file(uploaded_file)
                    st.session_state.vectorstore = v_store
                    st.session_state.raw_chunks = chunks
                    st.session_state.rag_chain = build_dynamic_rag_chain(st.session_state.vectorstore)
                    st.session_state.last_uploaded_name = uploaded_file.name
                    st.session_state.messages = []  # Clear historical conversation arrays
                    st.success("Document tracking established!")
                except Exception as ex:
                    st.error(f"Failed to process document layout: {ex}")

    if st.session_state.vectorstore is not None:
        st.markdown("---")
        st.subheader("📊 Live Evaluation Panel")
        st.caption("Generates unique verification criteria derived directly from this document's text structures.")
        
        eval_count = st.slider("Evaluation Sample Size", min_value=2, max_value=4, value=2)
        
        if st.button("🚀 Run Dynamic Ragas Evaluation", use_container_width=True):
            with st.spinner("Synthesizing metrics via Modern Decoupled Ragas Engine..."):
                from src.evaluation.evaluator import run_dynamic_evaluation
                results_df = run_dynamic_evaluation(
                    processed_documents=st.session_state.raw_chunks,
                    rag_chain=st.session_state.rag_chain,
                    vectorstore=st.session_state.vectorstore,
                    num_questions=eval_count
                )
                st.toast("Evaluation Matrix Complete!")
                st.dataframe(results_df)

    st.markdown("---")
    if st.button("🧹 Clear Conversation Window", use_container_width=True):
        st.session_state.messages = []
        st.rerun()



st.title("📚 Intelligent Document DeepDive Dashboard")
st.markdown("Interact dynamically with structural text contexts extracted through high-fidelity layout analytics engines.")
st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your uploaded document..."):
    if not st.session_state.rag_chain:
        st.warning("Please submit a target PDF asset via the sidebar to initialize operations.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            with status_placeholder.container():
                st.info("🔍 Step 1/3: Querying Direct Qdrant Dense+Sparse Vectors...")
                st.info("⚡ Step 2/3: Scoring context chunks via Jina CrossEncoder...")
                st.info("🧠 Step 3/3: Running generation via Gemini API...")
                
            try:
                import os
                from dotenv import load_dotenv
                from langfuse.langchain import CallbackHandler
                load_dotenv()
                
                langfuse_callback = None
                if os.environ.get('LANGFUSE_PUBLIC_KEY') and os.environ.get('LANGFUSE_SECRET_KEY'):
                    langfuse_callback = CallbackHandler()
                
                config = {}
                if langfuse_callback:
                    config["callbacks"] = [langfuse_callback]
                
                answer = st.session_state.rag_chain.invoke({"question": prompt}, config=config)
                status_placeholder.empty()
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                status_placeholder.empty()
                st.error(f"An operation processing failure occurred: {e}")

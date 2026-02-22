import streamlit as st
import os
from openai import OpenAI

from utils.file_loader import load_file
from rag.chunking import chunk_text
from rag.embeddings import embed_texts
from rag.retriever import build_faiss_index, retrieve
from rag.prompts import build_prompt, plan_query


# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "faiss_index": None,
        "metadata": [],
        "doc_names": [],
        "last_context": []
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─────────────────────────────────────────
# DOCUMENT PROCESSING PIPELINE
# ─────────────────────────────────────────
def process_documents(uploaded_files):
    """
    Full RAG ingestion pipeline:
    1. Load files
    2. Chunk text
    3. Embed chunks
    4. Build FAISS index
    5. Store everything in session state
    """
    all_chunks = []
    doc_names = []

    for file in uploaded_files:
        pages = load_file(file)

        for text, page_num in pages:
            chunks = chunk_text(text, page_num, doc_name=file.name)
            all_chunks.extend(chunks)

        doc_names.append(file.name)

    if not all_chunks:
        return

    # Embed in batch
    texts_only = [chunk["text"] for chunk in all_chunks]
    embeddings = embed_texts(texts_only)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Store in session state
    st.session_state.faiss_index = index
    st.session_state.metadata = all_chunks
    st.session_state.doc_names = doc_names


# ─────────────────────────────────────────
# RAG RESPONSE GENERATION
# ─────────────────────────────────────────
def generate_response(question: str) -> str:
    """Generate grounded answer using multi-query retrieval."""

    # Step 1 — Plan Retrieval Strategy
    plan = plan_query(question, client)
    sub_queries = plan.get("queries", [question])
    top_k = plan.get("top_k", 6)

    if question not in sub_queries:
        sub_queries.append(question)

    # Step 2 — Multi-Query Retrieval
    all_results = []

    for sub_query in sub_queries:
        query_embedding = embed_texts([sub_query])[0]

        results = retrieve(
            query_embedding,
            st.session_state.faiss_index,
            st.session_state.metadata,
            top_k=top_k
        )

        all_results.extend(results)

    # Step 3 — Deduplicate Results
    unique_chunks = {}
    for chunk in all_results:
        key = (chunk["text"], chunk["page"], chunk["doc_name"])
        unique_chunks[key] = chunk

    final_chunks = list(unique_chunks.values())
    st.session_state.last_context = final_chunks

    # Step 4 — Build Context
    context = "\n\n".join(
        f"[Page {r['page']} - {r['doc_name']}]\n{r['text']}"
        for r in final_chunks
    )

    # Step 5 — Build Final Prompt
    final_prompt = f"""
You are a document intelligence assistant.

Answer the user's question using ONLY the provided context.
If partial information exists, clearly explain conditions and limitations.
If information is missing, state that explicitly.

Context:
{context}

Question:
{question}
"""

    # Step 6 — Generate Answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You answer questions grounded strictly in provided document context."
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────
# CENTRAL QUESTION HANDLER
# ─────────────────────────────────────────
def handle_question(question: str):
    """Processes user or insight button queries."""
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.spinner("Thinking..."):
        answer = generate_response(question)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


# ─────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────
st.title("📄 DocuMind")
st.caption("🟢 Session-based | Documents deleted on refresh")
st.caption("🧠 RAG-powered document intelligence")
st.divider()

col1, col2 = st.columns([1, 2])


# ─────────────────────────────────────────
# LEFT COLUMN — DOCUMENTS & INSIGHTS
# ─────────────────────────────────────────
with col1:
    st.subheader("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        uploaded_names = [f.name for f in uploaded_files]

        if uploaded_names != st.session_state.doc_names:
            with st.spinner("Processing documents..."):
                process_documents(uploaded_files)
            st.success(f"✅ {len(uploaded_files)} document(s) ready")

    if st.session_state.doc_names:
        st.markdown("**Loaded Documents:**")
        for name in st.session_state.doc_names:
            st.markdown(f"- 📄 {name}")

    st.divider()

    # Quick Insights
    st.subheader("⚡ Quick Insights")
    st.caption("Requires a document to be uploaded first")

    INSIGHT_PROMPTS = {
        "📝 Summarize": "Summarize all uploaded documents clearly.",
        "🔑 Key Points": "What are the key points?",
        "⚠️ Risks": "What are potential risks or concerns?",
        "✅ Checklist": "Generate a checklist of action items."
    }

    for label, prompt in INSIGHT_PROMPTS.items():
        if st.button(label, use_container_width=True):
            if st.session_state.faiss_index is None:
                st.warning("Please upload a document first.")
            else:
                handle_question(prompt)


# ─────────────────────────────────────────
# RIGHT COLUMN — CHAT
# ─────────────────────────────────────────
with col2:
    st.subheader("💬 Chat")

    # Scrollable native container
    chat_container = st.container(height=520, border=True)

    with chat_container:
        if not st.session_state.messages:
            st.info("Upload a document and ask a question to get started.")
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    question = st.chat_input("Ask a question about your documents...")
    if question:
        if st.session_state.faiss_index is None:
            st.warning("Please upload a document first.")
        else:
            handle_question(question)
            st.rerun()


# ─────────────────────────────────────────
# RETRIEVED CONTEXT EXPANDER
# ─────────────────────────────────────────
st.divider()

with st.expander("🔍 Retrieved Context — Last Query"):
    if st.session_state.last_context:
        for i, chunk in enumerate(st.session_state.last_context):
            st.markdown(
                f"**Chunk {i+1} — Page {chunk['page']} · {chunk['doc_name']}**"
            )
            st.text(chunk["text"])
            st.divider()
    else:
        st.write("Retrieved chunks will appear here after your first question.")

st.caption("🔒 Files are processed in-memory and not stored anywhere.")
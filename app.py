"""
PDF Question Answering Agent — LangGraph + Groq + ChromaDB + Streamlit
-------------------------------------------------------------------------
Architecture:

    Upload PDF -> Read Content -> Split into Chunks -> Create Embeddings
        -> Store in Chroma Vector DB
        -> [ User asks Question ]
        -> LangGraph Workflow:  START -> retrieve -> generate -> END
        -> Groq (Llama) generates the answer
"""

import os
import json
import shutil
import tempfile
from datetime import datetime

from typing import TypedDict

import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


# ---------- Shared state for the LangGraph workflow ----------
class AgentState(TypedDict):
    question: str
    context: str
    context_chunks: list
    answer: str


# ---------- Persistence paths ----------
CHROMA_DIR = "./chroma_db"          # persistent vector store on disk
CHAT_LOG_PATH = "./chat_history.jsonl"  # persistent chat log on disk


# ---------- Persistent chat log helpers ----------
def load_chat_log():
    """Read all past Q&As from disk (survives app restarts)."""
    if not os.path.exists(CHAT_LOG_PATH):
        return []
    entries = []
    with open(CHAT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def append_chat_log(question: str, answer: str, pdf_name: str = ""):
    """Append one Q&A to the persistent log file."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pdf_name": pdf_name,
        "question": question,
        "answer": answer,
    }
    with open(CHAT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def clear_chat_log():
    if os.path.exists(CHAT_LOG_PATH):
        os.remove(CHAT_LOG_PATH)


# ---------- Embeddings model (cached so it only loads once per session) ----------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ---------- Step 1-4: Read PDF -> Chunk -> Embed -> Store in Chroma (persisted to disk) ----------
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(full_text)

    embeddings = get_embeddings()

    # Start fresh so this PDF's chunks don't mix with a previous PDF's chunks
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    vectordb = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    return vectordb, len(chunks)


def load_existing_index():
    """Load a previously-persisted Chroma index from disk without re-processing the PDF."""
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    try:
        count = vectordb._collection.count()
    except Exception:
        count = "?"
    return vectordb, count


# ---------- LangGraph workflow: START -> retrieve -> generate -> END ----------
def build_graph(vectordb):

    def retrieve_node(state: AgentState) -> AgentState:
        docs = vectordb.similarity_search(state["question"], k=3)
        state["context_chunks"] = [d.page_content for d in docs]
        state["context"] = "\n\n---\n\n".join(state["context_chunks"])
        return state

    def generate_node(state: AgentState) -> AgentState:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I couldn't find that in the document."

Context:
{state['context']}

Question: {state['question']}

Answer:"""
        response = llm.invoke([HumanMessage(content=prompt)])
        state["answer"] = response.content
        return state

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# ================= Streamlit UI =================

st.set_page_config(page_title="DocIntel · PDF Q&A", page_icon="📖", layout="centered")

# ---------- Design tokens & custom styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink: #0B0E14;
    --surface: #151920;
    --surface-2: #1B2029;
    --border: #262B36;
    --accent: #E8A33D;
    --accent-soft: rgba(232, 163, 61, 0.12);
    --evidence: #59C3B4;
    --evidence-soft: rgba(89, 195, 180, 0.10);
    --text: #E7E9EE;
    --muted: #8A93A3;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

h1, h2, h3 { font-family: 'Lora', serif !important; letter-spacing: -0.01em; }

/* Overall canvas */
[data-testid="stAppViewContainer"] > .main {
    background:
        radial-gradient(ellipse 900px 400px at 15% -10%, var(--accent-soft), transparent),
        var(--ink);
}
.block-container { padding-top: 2.5rem; max-width: 760px; }

/* Header */
.app-header-band {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.4rem;
    margin-bottom: 1.8rem;
}
.app-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.3rem;
}
.app-title {
    font-family: 'Lora', serif;
    font-weight: 600;
    font-size: 2.1rem;
    color: var(--text);
    margin: 0;
}
.app-subtitle {
    color: var(--muted);
    font-size: 0.92rem;
    margin-top: 0.35rem;
}
.app-subtitle .dot { color: var(--accent); margin: 0 0.5em; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
.sidebar-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.2rem;
}
section[data-testid="stSidebar"] h3 {
    color: var(--text) !important;
    font-size: 1.05rem !important;
    margin-bottom: 0.8rem;
}

/* Buttons (all, including "Browse files") */
.stButton button, [data-testid="stFileUploaderDropzone"] button, [data-testid="baseButton-secondary"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: 1px solid var(--border) !important;
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
    transition: all 0.15s ease;
}
.stButton button:hover, [data-testid="stFileUploaderDropzone"] button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
.stButton button[kind="primary"] {
    background-color: var(--accent) !important;
    color: #14110A !important;
    border: none !important;
}
.stButton button[kind="primary"]:hover {
    background-color: #f0b458 !important;
}

/* File uploader dropzone */
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--surface-2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploaderFile"] {
    background-color: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Inputs */
.stTextInput input {
    border-radius: 8px;
    border: 1px solid var(--border);
    background-color: var(--surface-2);
    color: var(--text);
}
.stTextInput input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
}

/* Alerts (info / success / error / warning) restyled to match the theme */
[data-testid="stAlertContainer"], .stAlert {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 4px solid var(--accent) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
[data-testid="stAlertContainer"] p, .stAlert p { color: var(--text) !important; }
[data-testid="stAlertContainer"] svg, .stAlert svg { display: none; }

/* Dividers */
[data-testid="stSidebar"] hr { border-color: var(--border); }

/* Answer card — the "highlighter on a page" signature element */
.answer-card {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-top: 0.6rem;
}
.answer-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
}
.answer-text {
    font-size: 1.02rem;
    line-height: 1.55;
    color: var(--text);
}

/* Evidence / retrieved chunks */
.evidence-chunk {
    background-color: var(--evidence-soft);
    border-left: 3px solid var(--evidence);
    border-radius: 6px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.5;
    color: #C9D4D1;
}
.evidence-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--evidence);
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
    display: block;
}

/* Status pill */
.status-pill {
    display: inline-block;
    background-color: var(--accent-soft);
    color: var(--accent);
    border: 1px solid rgba(232, 163, 61, 0.35);
    border-radius: 999px;
    padding: 0.3rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}

/* Chat history */
.chat-turn { margin-bottom: 1.1rem; }
.chat-q {
    color: var(--muted);
    font-size: 0.88rem;
    margin-bottom: 0.35rem;
}
.chat-q b { color: var(--text); font-weight: 600; }
.chat-a {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    color: var(--text);
    font-size: 0.95rem;
    line-height: 1.5;
}
.chat-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="app-header-band">
    <div class="app-eyebrow">RAG · LANGGRAPH WORKFLOW</div>
    <div class="app-title">📖 DocIntel — PDF Question Answering</div>
    <div class="app-subtitle">LangGraph<span class="dot">•</span>Groq (Llama)<span class="dot">•</span>ChromaDB<span class="dot">•</span>HuggingFace Embeddings</div>
</div>
""", unsafe_allow_html=True)

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "chat_history" not in st.session_state:
    # Loaded once per session from the persistent log, so old Q&As
    # show up again even after you restart the app.
    st.session_state.chat_history = load_chat_log()

with st.sidebar:
    st.markdown('<div class="sidebar-eyebrow">Configuration</div><h3>⚙️ Setup</h3>', unsafe_allow_html=True)

    api_key = st.text_input("Groq API Key", type="password",
                             help="Get a free key at https://console.groq.com")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.divider()

    st.markdown('<div class="sidebar-eyebrow">Knowledge Source</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and st.button("Process PDF", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your Groq API key first.")
        else:
            with st.spinner("Reading PDF, creating chunks, generating embeddings, storing in Chroma..."):
                vectordb, n_chunks = process_pdf(uploaded_file)
                st.session_state.vectordb = vectordb
                st.session_state.chunk_count = n_chunks
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.graph = build_graph(vectordb)
            st.success(f"Indexed {n_chunks} chunks. Ask questions below 👇")

    # Offer to reload a previously-persisted index without re-uploading/re-processing
    if not st.session_state.graph and os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        if st.button("📂 Load previously indexed PDF", use_container_width=True):
            with st.spinner("Loading existing index from disk..."):
                vectordb, count = load_existing_index()
                st.session_state.vectordb = vectordb
                st.session_state.chunk_count = count
                st.session_state.graph = build_graph(vectordb)
            st.success("Loaded existing index — no reprocessing needed!")

    if st.session_state.chunk_count:
        st.markdown(
            f'<div class="status-pill">📊 {st.session_state.chunk_count} chunks in Chroma</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<div class="sidebar-eyebrow">Chat History</div>', unsafe_allow_html=True)
    st.caption(f"{len(st.session_state.chat_history)} question(s) logged on disk")
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        clear_chat_log()
        st.rerun()

# ---------- Main panel: ask questions ----------
if st.session_state.graph:
    st.markdown("#### Ask a question about your document")
    question = st.text_input("Your question", placeholder="e.g. What is the main topic of this document?",
                              label_visibility="collapsed")

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving context and generating answer..."):
            result = st.session_state.graph.invoke(
                {"question": question, "context": "", "context_chunks": [], "answer": ""}
            )

        # 1. Session-only: keep it in st.session_state for this run
        # 2. Persistent: also append to the on-disk log so it survives restarts
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pdf_name": st.session_state.pdf_name,
            "question": question,
            "answer": result["answer"],
        }
        st.session_state.chat_history.append(entry)
        append_chat_log(question, result["answer"], st.session_state.pdf_name)

        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-label">💡 Answer</div>
            <div class="answer-text">{result["answer"]}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 See retrieved evidence (what the LLM actually read)"):
            for i, chunk in enumerate(result["context_chunks"], start=1):
                st.markdown(f"""
                <div class="evidence-chunk">
                    <span class="evidence-tag">EXCERPT {i}</span>{chunk}
                </div>
                """, unsafe_allow_html=True)

    # ---------- Scrolling chat history (persists across restarts) ----------
    if st.session_state.chat_history:
        st.markdown("#### 🕘 Chat history")
        for turn in reversed(st.session_state.chat_history):
            ts = turn.get("timestamp", "")
            st.markdown(f"""
            <div class="chat-turn">
                <div class="chat-q">🙋 <b>{turn['question']}</b></div>
                <div class="chat-a">{turn['answer']}</div>
                <div class="chat-meta">{ts}{' · ' + turn['pdf_name'] if turn.get('pdf_name') else ''}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("👈 Upload a PDF and click **Process PDF** in the sidebar to get started.")
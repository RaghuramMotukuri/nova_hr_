"""
app.py
LOVA HR — Production-Ready Streamlit Web Application
Clean rebuild with premium dark-mode UI and full pipeline integration.

Features:
  - 3 User-Selected Hugging Face Models (RoBERTa Extractive QA, Qwen 0.5B CausalLM, Flan-T5 Seq2Seq)
  - Top-Right "ℹ️ Pipeline Steps" Popover Info Button
  - Full-Width Clean Question & Answer UI
"""
import sys
import re
from pathlib import Path
from typing import Dict, List
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ROOT / ".env", override=True)
except ImportError:
    pass

# ── Streamlit page config (MUST be first st call) ─────────────────────────────
st.set_page_config(
    page_title="LOVA HR — AI Policy Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Dark-Mode CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary:    #0a0f1e;
    --bg-surface:    #0f1629;
    --bg-elevated:   #151f3a;
    --bg-card:       #1a2545;
    --accent-blue:   #4f7cf7;
    --accent-indigo: #6366f1;
    --accent-cyan:   #06b6d4;
    --accent-emerald:#10b981;
    --accent-rose:   #f43f5e;
    --accent-amber:  #f59e0b;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #64748b;
    --border:        rgba(99,102,241,0.2);
    --border-hover:  rgba(79,124,247,0.5);
    --glow:          0 0 30px rgba(79,124,247,0.15);
    --radius:        12px;
    --radius-sm:     8px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.app-header-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, #0f1629 0%, #1a2545 50%, #0f1629 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.2rem 2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    position: relative; overflow: hidden;
}
.app-header-container::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(79,124,247,0.08) 0%, transparent 70%);
}
.app-header-left {
    display: flex; align-items: center; gap: 1rem;
}
.app-header-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #4f7cf7, #6366f1);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; box-shadow: 0 0 20px rgba(79,124,247,0.4);
    flex-shrink: 0;
}
.app-header-title { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; }
.app-header-sub   { font-size: 0.78rem; color: var(--text-secondary); margin-top: 2px; }

[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

.sidebar-section {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem; margin-bottom: 0.75rem;
}
.sidebar-section-title {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6366f1 !important;
    margin-bottom: 0.6rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

.status-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.25rem 0.7rem; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600;
}
.status-online  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.status-offline { background: rgba(244,63,94,0.15);  color: #f43f5e; border: 1px solid rgba(244,63,94,0.3); }
.status-warning { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }

.stTextArea textarea, .stTextInput input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(79,124,247,0.15) !important;
}

.answer-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem; margin: 1rem 0;
    box-shadow: var(--glow);
    position: relative; overflow: hidden;
}
.answer-card::before {
    content: "";
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #4f7cf7, #6366f1);
}
.answer-card-header {
    display: flex; align-items: center; gap: 0.6rem;
    margin-bottom: 1rem; padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}
.provider-badge {
    padding: 0.25rem 0.8rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.03em;
}
.answer-text { line-height: 1.8; color: var(--text-primary); font-size: 0.95rem; }

.citation-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem; margin-bottom: 0.5rem;
}

.metric-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem; text-align: center;
}
.metric-value {
    font-size: 1.8rem; font-weight: 800;
    background: linear-gradient(135deg, #4f7cf7, #6366f1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-label { font-size: 0.72rem; color: var(--text-secondary); margin-top: 2px; }

/* Multi-Company Comparison Cards */
.multi-company-general {
    background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.04) 100%);
    border: 1px solid rgba(16,185,129,0.3);
    border-left: 4px solid #10b981;
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
}
.multi-company-general .answer-card-header {
    border-bottom-color: rgba(16,185,129,0.2);
}
.multi-company-diff {
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(99,102,241,0.04) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-left: 4px solid #6366f1;
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
}
.multi-company-diff .answer-card-header {
    border-bottom-color: rgba(99,102,241,0.2);
}
.multi-company-summary {
    background: linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(245,158,11,0.04) 100%);
    border: 1px solid rgba(245,158,11,0.3);
    border-left: 4px solid #f59e0b;
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
}
.multi-company-summary .answer-card-header {
    border-bottom-color: rgba(245,158,11,0.2);
}
.company-tag {
    display: inline-block;
    background: rgba(79,124,247,0.12);
    color: var(--accent-blue);
    border: 1px solid rgba(79,124,247,0.3);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.7rem;
    font-weight: 600;
    margin: 0.2rem;
}
.multi-company-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(16,185,129,0.12);
    color: #10b981;
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 600;
}

/* Sticky Top Header Panel */
.sticky-top-panel {
    position: sticky;
    top: 0;
    z-index: 9999;
    background: linear-gradient(135deg, #0a0f1e 0%, #0f1629 50%, #0a0f1e 100%);
    border-bottom: 1px solid var(--border);
    padding: 0.75rem 1.5rem;
    margin: -1rem -1rem 0.5rem -1rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    backdrop-filter: blur(12px);
}
.sticky-top-panel .app-header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.sticky-top-panel .app-header-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #4f7cf7, #6366f1);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 0 15px rgba(79,124,247,0.4);
    flex-shrink: 0;
}
.sticky-top-panel .app-header-title {
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
}
.sticky-top-panel .app-header-sub {
    font-size: 0.65rem;
    color: var(--text-secondary);
    margin-top: 1px;
}
.sticky-top-panel [data-testid="stHorizontalBlock"] {
    background: transparent !important;
}
.sticky-top-panel [data-testid="stHorizontalBlock"] > div {
    background: transparent !important;
}
/* Tab strip inside sticky panel */
.sticky-tabs {
    position: sticky;
    top: 85px;
    z-index: 9998;
    background: rgba(10,15,30,0.95);
    border-bottom: 1px solid var(--border);
    padding: 0.25rem 0.5rem;
    margin: 0 -1rem;
    backdrop-filter: blur(8px);
}
.sticky-tabs [data-testid="stTabs"] {
    border: none !important;
    background: transparent !important;
}
.sticky-tabs [data-testid="stTabs"] > div[role="tablist"] {
    background: transparent !important;
    border-bottom: none !important;
}

.stButton > button {
    background: linear-gradient(135deg, #4f7cf7, #6366f1) !important;
    color: #fff !important; border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    box-shadow: 0 4px 15px rgba(79,124,247,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(79,124,247,0.4) !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, rgba(79,124,247,0.2), rgba(99,102,241,0.2)) !important;
    border-bottom: 2px solid #4f7cf7 !important;
    color: #4f7cf7 !important;
}

hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-surface); }
::-webkit-scrollbar-thumb { background: rgba(79,124,247,0.5); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Sticky Top Panel (Header + Tabs) ─────────────────────────────────────────
st.markdown('<div class="sticky-top-panel">', unsafe_allow_html=True)

col_head_left, col_head_right = st.columns([4, 1], vertical_alignment="center")

with col_head_left:
    st.markdown("""
    <div class="app-header-left">
        <div class="app-header-icon">🧠</div>
        <div>
            <div class="app-header-title">LOVA HR — AI Policy Assistant</div>
            <div class="app-header-sub">
                BGE-Large Embeddings &bull; BGE Reranker V2 M3 &bull; RoBERTa / Qwen 0.5B / Flan-T5 &bull; Firebase Vector Search
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_head_right:
    with st.popover("ℹ️ About Pipeline", use_container_width=True):
        st.markdown("### 💡 Pipeline Steps")
        steps = [
            ("1", "Sanitize",    "Strip index markers from query and context", "#4f7cf7"),
            ("2", "Boundary",    "Topic boundary check — refuse out-of-scope", "#f43f5e"),
            ("3", "BM25",        "Keyword search over indexed policy chunks",   "#06b6d4"),
            ("4", "Semantic",    "Firestore vector search (BGE-Large embeddings)", "#6366f1"),
            ("5", "RRF Fusion",  "Reciprocal Rank Fusion merges both signals",  "#10b981"),
            ("6", "Reranker",    "BGE Reranker V2 M3 cross-encoder scoring",   "#f59e0b"),
            ("7", "LLM Answer",  "Selected HF model grounded answer generation", "#ec4899"),
        ]
        for num, title, desc, color in steps:
            st.markdown(f"""
<div style="display:flex;gap:0.75rem;margin-bottom:0.5rem;padding:0.6rem;
            background:#1a2545;border-radius:8px;border-left:3px solid {color};">
    <div style="background:{color}22;color:{color};min-width:20px;height:20px;
                border-radius:5px;display:flex;align-items:center;justify-content:center;
                font-size:0.7rem;font-weight:700;">{num}</div>
    <div>
        <div style="font-weight:600;font-size:0.82rem;color:#f1f5f9;">{title}</div>
        <div style="font-size:0.72rem;color:#64748b;">{desc}</div>
    </div>
</div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "retriever": None, "pipeline": None,
    "chat_history": [], "docs_indexed": 0, "last_result": None,
    "confirm_del_company": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing AI pipeline...")
def get_retriever():
    from src.retriever import HybridRetriever
    from src.embeddings import EmbeddingPipeline
    pipeline = EmbeddingPipeline()
    retriever = HybridRetriever(pipeline=pipeline)
    try:
        from src.data_loader import load_all_documents
        docs = load_all_documents("data")
        if docs:
            pipeline.add_documents(docs)
            print(f"[App] Pre-loaded {len(docs)} document chunks from data/")
    except Exception as exc:
        print(f"[App] Data pre-load warning: {exc}")
    return retriever


@st.cache_data(show_spinner=False, ttl=300)
def check_firebase():
    from src.config import is_firebase_available
    return is_firebase_available()


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"
    return "79,124,247"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # New Chat at top
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">🆕 New Chat</div>', unsafe_allow_html=True)
    if st.button("➕ Start New Conversation", key="new_chat_btn", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.success("Started a new conversation.")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Firebase status
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">🔥 Firebase Status</div>', unsafe_allow_html=True)
    fb_ok = check_firebase()
    badge_class = "status-online" if fb_ok else "status-offline"
    badge_text  = "● Online — Semantic Search Active" if fb_ok else "● Offline — BM25 Only Mode"
    st.markdown(f'<span class="status-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Tenant
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">🏢 Tenant Scope</div>', unsafe_allow_html=True)
    company_options = ["All Companies"]
    data_path = ROOT / "data"
    if data_path.exists():
        subdirs = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        company_options.extend(subdirs)
    selected_company = st.selectbox(
        "Company", company_options, key="company_select", label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Model & RAG (ONLY the 3 requested HuggingFace models)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">🤖 AI Model Settings</div>', unsafe_allow_html=True)
    from src.generator import HF_MODELS, DEFAULT_HF_MODEL

    model_keys   = list(HF_MODELS.keys())
    model_labels = [f"{HF_MODELS[m]['icon']} {HF_MODELS[m]['label']}" for m in model_keys]
    default_idx  = model_keys.index(DEFAULT_HF_MODEL) if DEFAULT_HF_MODEL in model_keys else 0

    selected_label = st.selectbox(
        "HF Model", model_labels, index=default_idx, key="model_select", label_visibility="collapsed"
    )
    selected_model = model_keys[model_labels.index(selected_label)]

    model_desc = HF_MODELS.get(selected_model, {}).get("description", "")
    if model_desc:
        st.markdown(f"<div style='font-size:0.72rem;color:#64748b;margin-top:0.25rem;'>{model_desc}</div>",
                    unsafe_allow_html=True)

    rag_mode = st.radio(
        "RAG Mode",
        ["hybrid", "semantic", "fullylexical"],
        format_func=lambda x: {
            "hybrid":       "🔀 Hybrid (Recommended)",
            "semantic":     "🔵 Semantic Only",
            "fullylexical": "🔤 Lexical Only",
        }[x],
        key="rag_mode_radio",
    )

    use_local = st.toggle(
        "Use Local Transformers",
        value=True,
        key="local_engine",
        help="Run locally via transformers (no cloud API needed).",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Search params
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">⚙️ Search Parameters</div>', unsafe_allow_html=True)
    top_k        = st.slider("Top Results (K)", 1, 10, 5, key="top_k_slider")
    rerank_top_n = st.slider("Rerank Top N",    1, 10, 5, key="rerank_n_slider")
    st.markdown("</div>", unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">📁 Document Upload</div>', unsafe_allow_html=True)
    _upload_companies = (
        sorted([d.name for d in (ROOT / "data").iterdir() if d.is_dir()])
        if (ROOT / "data").exists() else []
    )
    if not _upload_companies:
        _upload_companies = ["General"]
    _up_default = selected_company if selected_company in _upload_companies else _upload_companies[0]
    upload_company = st.selectbox(
        "Upload to Company", _upload_companies,
        index=_upload_companies.index(_up_default), key="upload_company_select",
    )
    uploaded = st.file_uploader(
        "Upload HR Policies", type=["pdf", "txt", "docx"],
        accept_multiple_files=True, key="doc_uploader", label_visibility="collapsed",
    )
    if uploaded and st.button("📥 Index Documents", key="index_btn", use_container_width=True):
        with st.spinner("Indexing documents..."):
            try:
                retriever = get_retriever()
                all_docs = []
                tenant = upload_company
                upload_dir = ROOT / "data" / tenant
                upload_dir.mkdir(parents=True, exist_ok=True)
                for f in uploaded:
                    dest = upload_dir / f.name
                    dest.write_bytes(f.getvalue())
                from src.data_loader import _load_pdfs, _load_txts, _load_docx
                all_docs += _load_pdfs(upload_dir)
                all_docs += _load_txts(upload_dir)
                all_docs += _load_docx(upload_dir)
                if all_docs:
                    retriever.pipeline.add_documents(all_docs)
                    st.session_state.docs_indexed += len(all_docs)
                    store = retriever.pipeline.vector_store
                    if store.is_online:
                        st.success(
                            f"Indexed {len(all_docs)} chunks for '{tenant}' — saved to "
                            f"data/{tenant}/ and synced to Firestore."
                        )
                    else:
                        st.warning(
                            f"Indexed {len(all_docs)} chunks for '{tenant}' locally in "
                            f"data/{tenant}/. Firebase is offline — use 'Sync to Firebase' "
                            "to upload them later."
                        )
                else:
                    st.warning("No processable content found.")
            except Exception as exc:
                st.error(f"Indexing error: {exc}")

    if st.button("🔄 Sync Local Files to Firebase", key="sync_btn", use_container_width=True):
        with st.spinner("Syncing to Firebase..."):
            try:
                retriever = get_retriever()
                store = retriever.pipeline.vector_store
                if not store.is_online:
                    st.warning("Firebase is offline — cannot sync to Firestore right now.")
                else:
                    chunks = store.get_all_chunks()
                    store.upsert_chunks(chunks)
                    st.success(f"Synced {len(chunks)} chunks to Firestore.")
            except Exception as exc:
                st.error(f"Sync error: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Stats
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">📊 Session Stats</div>', unsafe_allow_html=True)
    st.metric("Queries", len(st.session_state.chat_history))
    st.metric("Indexed Chunks", st.session_state.docs_indexed)
    if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_result  = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ── Main tabs ─────────────────────────────────────────────────────────────────
def render_chunks(tab, chunks, max_n=10):
    """Render a list of retrieval chunks inside a Streamlit tab."""
    with tab:
        if not chunks:
            st.info("No results in this category.")
            return
        for i, chunk in enumerate(chunks[:max_n], 1):
            score = chunk.get("reranker_score", chunk.get("rrf_score", chunk.get("bm25_score", 0)))
            s_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            with st.expander(f"#{i} — {chunk.get('filename','?')} p.{chunk.get('page_number','')} (Score: {s_str})"):
                st.markdown(f"**Company:** {chunk.get('company','')}")
                st.markdown(f"**Section:** {chunk.get('section_header', chunk.get('subtopic',''))}")
                st.markdown("---")
                st.markdown(chunk.get("chunk_text", ""))


# ── Sticky Tab Strip ─────────────────────────────────────────────────────────
st.markdown('<div class="sticky-tabs">', unsafe_allow_html=True)
tab_search, tab_history, tab_docs = st.tabs(["🔍 Search", "🕘 History", "📚 Documents"])
st.markdown('</div>', unsafe_allow_html=True)


# ── Search tab (Chat Interface) ────────────────────────────────────────────────
with tab_search:
    # Render full chat history
    for turn in st.session_state.chat_history:
        # User message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(turn["query"])

        # Assistant message
        result = turn["result"]
        gen = result.get("generation", {})
        answers = gen.get("answers", [])
        citations = gen.get("citations", [])
        is_multi_company = gen.get("multi_company", False)

        with st.chat_message("assistant", avatar="🤖"):
            if is_multi_company:
                # ── Multi-Company Comparison Format ────────────────────────
                companies = gen.get("companies", [])
                company_groups = gen.get("company_groups", {})
                
                # Multi-company badge
                company_tags = " ".join([
                    f'<span class="company-tag">{c}</span>' for c in companies[:5]
                ])
                st.markdown(
                    f'<div class="multi-company-badge">📊 Comparing {len(companies)} companies</div>'
                    f'<div style="margin: 0.5rem 0;">{company_tags}</div>',
                    unsafe_allow_html=True,
                )
                
                # Render each section with specific styling
                for ans in answers:
                    bc = ans.get("badge_color", "#6366f1")
                    icon = ans.get("icon", "🤖")
                    label = ans.get("label", "Policy Processor")
                    section_type = ans.get("section_type", "general")
                    text = ans.get("answer", "").replace("\n", "<br>")
                    rgb = _hex_to_rgb(bc)
                    
                    # Use specific CSS class based on section type
                    card_class = {
                        "general": "multi-company-general",
                        "differences": "multi-company-diff",
                        "summary": "multi-company-summary",
                    }.get(section_type, "answer-card")
                    
                    st.markdown(f"""
<div class="{card_class}">
    <div class="answer-card-header">
        <span style="font-size:1.2rem;">{icon}</span>
        <span class="provider-badge"
              style="background:rgba({rgb},0.15);color:{bc};border:1px solid {bc}40;">
            {label}
        </span>
    </div>
    <div class="answer-text">{text}</div>
</div>""", unsafe_allow_html=True)
                
                # Company groups in expandable sections
                if company_groups:
                    with st.expander("🏢 Company Policy Details", expanded=False):
                        for company, comp_chunks in company_groups.items():
                            st.markdown(f"**{company}**")
                            for chunk in comp_chunks[:2]:
                                chunk_text = chunk.get("chunk_text", "")[:500]
                                st.markdown(f"""
<div class="citation-card">
    <div style="font-size:0.82rem;color:#94a3b8;">
        {chunk.get('filename', 'Policy')} — p.{chunk.get('page_number', '')}
    </div>
    <div style="font-size:0.85rem;margin-top:0.4rem;">{chunk_text}</div>
</div>""", unsafe_allow_html=True)
                            st.markdown("---")
            
            else:
                # ── Standard Single-Company Format ────────────────────────
                for ans in answers:
                    bc = ans.get("badge_color", "#6366f1")
                    icon = ans.get("icon", "🤖")
                    label = ans.get("label", "Policy Processor")
                    text = ans.get("answer", "").replace("\n", "<br>")
                    rgb = _hex_to_rgb(bc)
                    st.markdown(f"""
<div class="answer-card">
    <div class="answer-card-header">
        <span style="font-size:1.2rem;">{icon}</span>
        <span class="provider-badge"
              style="background:rgba({rgb},0.15);color:{bc};border:1px solid {bc}40;">
            {label}
        </span>
    </div>
    <div class="answer-text">{text}</div>
</div>""", unsafe_allow_html=True)

            if citations:
                with st.expander(f"📚 Source Citations ({len(citations)})", expanded=True):
                    for cit in citations:
                        score = cit.get("score", 0)
                        score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
                        fname = cit.get("filename", "Unknown")
                        page = cit.get("page", "")
                        sec = cit.get("section_header", "")
                        rank = cit.get("rank", "?")
                        st.markdown(f"""
<div class="citation-card">
    <strong>#{rank} — {fname}</strong>
    <span style="color:#94a3b8;font-size:0.8rem;margin-left:0.5rem;">
        p.{page} &bull; {sec} &bull; Score: {score_str}
    </span>
</div>""", unsafe_allow_html=True)

            if result.get("mode") == "bm25_only":
                st.markdown(
                    '<span class="status-badge status-warning">'
                    "⚠️ BM25-only — Configure Firebase for full semantic search"
                    "</span>",
                    unsafe_allow_html=True,
                )

            # Detailed retrieval results
            with st.expander("🔍 Detailed Results", expanded=False):
                detail_r, detail_l, detail_s = st.tabs(["🔀 Reranked", "🔤 Lexical", "🔵 Semantic"])
                render_chunks(detail_r, result.get("reranked", []))
                render_chunks(detail_l, result.get("lexical", []))
                render_chunks(detail_s, result.get("semantic", []))


# ── History tab (all conversations) ───────────────────────────────────────────
with tab_history:
    st.markdown("### 🕘 Conversation History")
    history = st.session_state.chat_history
    if not history:
        st.info("No conversations yet. Ask a question from the Search tab.")
    else:
        st.caption(f"{len(history)} saved question(s) in this conversation.")
        for i, turn in enumerate(reversed(history)):
            with st.expander(f"#{len(history) - i} — {turn['query'][:70]}{'…' if len(turn['query']) > 70 else ''}", expanded=False):
                st.markdown(f"**❓ {turn['query']}**")
                st.markdown("---")
                result = turn["result"]
                gen = result.get("generation", {})
                answers = gen.get("answers", [])
                citations = gen.get("citations", [])
                is_multi_company = gen.get("multi_company", False)

                if is_multi_company:
                    companies = gen.get("companies", [])
                    company_tags = " ".join([
                        f'<span class="company-tag">{c}</span>' for c in companies[:5]
                    ])
                    st.markdown(
                        f'<div class="multi-company-badge">📊 Comparing {len(companies)} companies</div>'
                        f'<div style="margin: 0.5rem 0;">{company_tags}</div>',
                        unsafe_allow_html=True,
                    )

                for ans in answers:
                    bc = ans.get("badge_color", "#6366f1")
                    icon = ans.get("icon", "🤖")
                    label = ans.get("label", "Policy Processor")
                    section_type = ans.get("section_type", "general")
                    text = ans.get("answer", "").replace("\n", "<br>")
                    rgb = _hex_to_rgb(bc)
                    
                    card_class = {
                        "general": "multi-company-general",
                        "differences": "multi-company-diff",
                        "summary": "multi-company-summary",
                    }.get(section_type, "answer-card") if is_multi_company else "answer-card"
                    
                    st.markdown(f"""
<div class="{card_class}">
    <div class="answer-card-header">
        <span style="font-size:1.2rem;">{icon}</span>
        <span class="provider-badge"
              style="background:rgba({rgb},0.15);color:{bc};border:1px solid {bc}40;">
            {label}
        </span>
    </div>
    <div class="answer-text">{text}</div>
</div>""", unsafe_allow_html=True)

                if citations:
                    st.markdown("**📚 Sources**")
                    for cit in citations:
                        score = cit.get("score", 0)
                        score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
                        fname = cit.get("filename", "Unknown")
                        page = cit.get("page", "")
                        sec = cit.get("section_header", "")
                        rank = cit.get("rank", "?")
                        st.markdown(f"""
<div class="citation-card">
    <strong>#{rank} — {fname}</strong>
    <span style="color:#94a3b8;font-size:0.8rem;margin-left:0.5rem;">
        p.{page} &bull; {sec} &bull; Score: {score_str}
    </span>
</div>""", unsafe_allow_html=True)

                with st.expander("🔍 Detailed Results", expanded=False):
                    detail_r, detail_l, detail_s = st.tabs(["🔀 Reranked", "🔤 Lexical", "🔵 Semantic"])
                    render_chunks(detail_r, result.get("reranked", []))
                    render_chunks(detail_l, result.get("lexical", []))
                    render_chunks(detail_s, result.get("semantic", []))

                if st.button("🗑️ Delete", key=f"hist_del_{i}", use_container_width=True):
                    del st.session_state.chat_history[len(history) - 1 - i]
                    if not st.session_state.chat_history:
                        st.session_state.last_result = None
                    st.rerun()


# ── Documents tab ─────────────────────────────────────────────────────────────
def _delete_document(fpath: Path) -> None:
    """Delete a file on disk and all of its indexed chunks (memory + Firestore)."""
    try:
        retriever = get_retriever()
        retriever.pipeline.vector_store.delete_file_chunks(str(fpath.resolve()))
        fpath.unlink(missing_ok=True)
        # Rebuild the in-memory BM25 index without the deleted chunks
        try:
            chunks = retriever.pipeline.vector_store.get_all_chunks()
            retriever.pipeline.build_bm25_index(chunks)
        except Exception as exc:
            print(f"[App] BM25 rebuild warning after delete: {exc}")
        st.success(f"Deleted '{fpath.name}' and its indexed chunks.")
        st.rerun()
    except Exception as exc:
        st.error(f"Delete failed: {exc}")


def _delete_company(company: str) -> None:
    """Delete a company: all its files on disk and every indexed chunk (memory + Firestore)."""
    try:
        retriever = get_retriever()
        retriever.pipeline.vector_store.delete_company_chunks(company)
        company_dir = ROOT / "data" / company
        if company_dir.exists() and company_dir.is_dir():
            for fpath in company_dir.rglob("*"):
                if fpath.is_file():
                    fpath.unlink(missing_ok=True)
            for sub in sorted(company_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if sub.is_dir():
                    try:
                        sub.rmdir()
                    except OSError:
                        pass
            try:
                company_dir.rmdir()
            except OSError:
                pass
        try:
            chunks = retriever.pipeline.vector_store.get_all_chunks()
            retriever.pipeline.build_bm25_index(chunks)
        except Exception as exc:
            print(f"[App] BM25 rebuild warning after company delete: {exc}")
        st.success(f"Deleted company '{company}' and all its files and indexed chunks.")
    except Exception as exc:
        st.error(f"Delete company failed: {exc}")


with tab_docs:
    st.markdown("### 📁 Document Library")
    data_path = ROOT / "data"
    if data_path.exists():
        companies = sorted([d.name for d in data_path.iterdir() if d.is_dir()])

        # ── Create a new company folder ──
        with st.expander("➕ Create New Company Folder", expanded=False):
            nc_col1, nc_col2 = st.columns([3, 1])
            with nc_col1:
                new_company = st.text_input(
                    "Company name", key="new_company_input", label_visibility="collapsed",
                    placeholder="e.g., Acme Corp",
                )
            with nc_col2:
                if st.button("Create", key="create_company_btn", use_container_width=True):
                    name = new_company.strip()
                    if not name:
                        st.error("Enter a company name first.")
                    elif not re.fullmatch(r"[A-Za-z0-9 _\-]+", name):
                        st.error("Company name may only contain letters, numbers, spaces, hyphens, and underscores.")
                    elif (data_path / name).exists():
                        st.error(f"Company '{name}' already exists.")
                    else:
                        (data_path / name).mkdir(parents=True, exist_ok=True)
                        st.success(
                            f"Company folder '{name}' created. Select it in the sidebar "
                            "Tenant Scope and upload documents to tag them under this company."
                        )
                        st.rerun()

        # ── Delete a company (removes all its files + indexed chunks) ──
        with st.expander("🗑️ Delete Company", expanded=False):
            del_target = st.selectbox(
                "Company to delete", ["— Select —"] + companies, key="del_company_select"
            )
            if del_target != "— Select —" and st.button(
                "Delete Company", key="del_company_btn", use_container_width=True
            ):
                st.session_state.confirm_del_company = del_target
                st.rerun()
            if st.session_state.get("confirm_del_company") in companies:
                target = st.session_state.confirm_del_company
                st.warning(
                    f"⚠️ Delete company '{target}' and ALL of its files and indexed "
                    "chunks? This cannot be undone."
                )
                cf_col1, cf_col2 = st.columns(2)
                with cf_col1:
                    if st.button("Yes, delete", key="confirm_del_company_yes", use_container_width=True):
                        _delete_company(target)
                        st.session_state.confirm_del_company = None
                        st.rerun()
                with cf_col2:
                    if st.button("Cancel", key="confirm_del_company_no", use_container_width=True):
                        st.session_state.confirm_del_company = None
                        st.rerun()

        # ── Company filter (defaults to the sidebar tenant selection) ──
        company_options = ["All Companies"] + companies
        default_idx = company_options.index(selected_company) if selected_company in company_options else 0
        filter_company = st.selectbox(
            "🏢 Company", company_options, index=default_idx, key="docs_company_filter"
        )

        # ── File-name search ──
        search_term = st.text_input(
            "🔎 Find file", key="docs_search", label_visibility="collapsed",
            placeholder="Search files by name...",
        ).strip().lower()

        all_files = (
            list(data_path.glob("**/*.pdf")) +
            list(data_path.glob("**/*.txt")) +
            list(data_path.glob("**/*.docx"))
        )
        all_files = [f for f in all_files if not f.name.startswith(("~$", ".", "__"))]
        total_files = len(all_files)

        if filter_company != "All Companies":
            all_files = [
                f for f in all_files
                if (f.relative_to(data_path).parts[0] if len(f.relative_to(data_path).parts) > 1 else "General") == filter_company
            ]
        if search_term:
            all_files = [f for f in all_files if search_term in f.name.lower()]

        if total_files == 0:
            st.markdown("""
<div style="text-align:center;padding:3rem;color:#64748b;">
    <div style="font-size:3rem;margin-bottom:1rem;">📂</div>
    <div style="font-weight:600;color:#94a3b8;">No documents in data/</div>
    <div style="font-size:0.85rem;margin-top:0.5rem;">
        Upload via the sidebar or place HR policy files in the data/ folder.
    </div>
</div>""", unsafe_allow_html=True)
        elif not all_files:
            st.info("No files match the current filters. Adjust the company or search term.")
        else:
            # Group files by company so the library reads as company categories
            by_company: Dict[str, List[Path]] = {}
            for fpath in all_files:
                rel = fpath.relative_to(data_path)
                company = rel.parts[0] if len(rel.parts) > 1 else "General"
                by_company.setdefault(company, []).append(fpath)

            ext_icons = {"pdf": "📄", "txt": "📝", "docx": "📋"}
            for company in sorted(by_company):
                st.markdown(f"### 🏢 {company}")
                cols = st.columns(3)
                for i, fpath in enumerate(by_company[company]):
                    with cols[i % 3]:
                        rel      = fpath.relative_to(data_path)
                        parts    = rel.parts
                        subfolder = "/".join(parts[1:-1]) if len(parts) > 2 else "General"
                        size_kb  = fpath.stat().st_size // 1024
                        icon     = ext_icons.get(fpath.suffix.lstrip("."), "📁")
                        st.markdown(f"""
<div class="citation-card">
    <div style="font-size:1.3rem;">{icon}</div>
    <div style="font-weight:600;font-size:0.85rem;margin-top:4px;">{fpath.name}</div>
    <div style="font-size:0.75rem;color:#94a3b8;">{subfolder}</div>
    <div style="font-size:0.72rem;color:#64748b;">{size_kb} KB</div>
</div>""", unsafe_allow_html=True)
                        if st.button("🗑️ Delete", key=f"del_{rel}", use_container_width=True):
                            _delete_document(fpath)
    else:
        st.info("data/ directory not found. Create it and add HR policy documents.")


# ── Floating Chat Input (Bottom Panel) ────────────────────────────────────────
with st.bottom:
    st.markdown("""
    <style>
    .floating-chat-panel {
        background: linear-gradient(135deg, #0f1629 0%, #1a2545 100%);
        border-top: 1px solid var(--border);
        padding: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    user_query = st.chat_input(
        "Ask a policy question... (Enter to send, Shift+Enter for new line)",
        key="floating_chat_input",
    )
    
    if user_query:
        with st.spinner("Searching policy documents..."):
            try:
                retriever = get_retriever()
                company_arg = None if selected_company == "All Companies" else selected_company
                result = retriever.search(
                    query=user_query,
                    top_k=top_k,
                    company=company_arg,
                    rerank_top_n=rerank_top_n,
                    rag_mode=rag_mode,
                    hf_model=selected_model,
                    use_local_engine=use_local,
                )
                st.session_state.last_result = result
                st.session_state.chat_history.append({"query": user_query, "result": result})
                st.rerun()
            except Exception as exc:
                st.error(f"Search error: {exc}")

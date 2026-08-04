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
import time
from pathlib import Path
from typing import Dict, List
import streamlit as st

import atexit
import shutil


class Timer:
    """Context manager for timing operations and logging to terminal."""
    def __init__(self, label: str):
        self.label = label
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f"[Timer] {self.label}: {self.elapsed:.3f}s")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def _cleanup_session_temp_files():
    """Ensure temporary local chunk files in data/ are wiped on app process termination."""
    try:
        data_dir = ROOT / "data"
        if data_dir.exists():
            for item in data_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                elif item.is_file() and not item.name.startswith("."):
                    item.unlink(missing_ok=True)
    except Exception as exc:
        print(f"[App] Session cleanup notice: {exc}")

atexit.register(_cleanup_session_temp_files)

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
.company-highlight {
    background: rgba(79,124,247,0.12);
    color: #4f7cf7;
    border: 1px solid rgba(79,124,247,0.3);
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    font-weight: 600;
    font-size: 0.9em;
}
.company-sentence-break { margin-top: 0.6rem; display: block; }

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
    "docs_indexed": 0, "last_result": None,
    "chat_history": [],
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
    from src.firebase_client import FAISSIndex
    pipeline = EmbeddingPipeline()
    retriever = HybridRetriever(pipeline=pipeline)
    
    # Initialize FAISS index for fast vector search
    faiss_index = FAISSIndex()
    retriever.faiss_index = faiss_index
    
    # Load embeddings from Firestore into FAISS on startup
    if pipeline.vector_store.is_online:
        try:
            count = faiss_index.load_from_firestore(pipeline.vector_store)
            print(f"[App] FAISS index loaded: {count} vectors")
        except Exception as exc:
            print(f"[App] FAISS load notice: {exc}")
        
        # Pre-build BM25 index from Firestore for instant search
        try:
            all_chunks = pipeline.vector_store.get_all_documents()
            if all_chunks:
                pipeline.build_bm25_index(all_chunks)
                print(f"[App] BM25 index pre-built: {len(all_chunks)} chunks")
        except Exception as exc:
            print(f"[App] BM25 pre-build notice: {exc}")
    
    return retriever


def _preload_local_model():
    """Pre-load local model in background thread for instant first query."""
    import threading
    def _load():
        try:
            from src.generator import LLMGenerator
            gen = LLMGenerator()
            gen._load_local_model("Qwen/Qwen2.5-0.5B-Instruct")
            print("[App] Local model pre-loaded successfully")
        except Exception as exc:
            print(f"[App] Model pre-load notice: {exc}")
    thread = threading.Thread(target=_load, daemon=True)
    thread.start()


# Pre-load local model on startup (background thread)
if "model_preloaded" not in st.session_state:
    st.session_state.model_preloaded = True
    _preload_local_model()


@st.cache_data(show_spinner=False, ttl=300)
def check_firebase():
    from src.config import is_firebase_available
    return is_firebase_available()


def _ensure_data_loaded(retriever):
    """Lazy-load documents on first search if not already indexed."""
    if st.session_state.docs_indexed > 0:
        return
    try:
        from src.data_loader import load_all_documents
        docs = load_all_documents("data")
        if docs:
            retriever.pipeline.add_documents(docs)
            st.session_state.docs_indexed = len(docs)
            print(f"[App] Lazy-loaded {len(docs)} document chunks from data/")
    except Exception as exc:
        print(f"[App] Data lazy-load notice: {exc}")


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"
    return "79,124,247"


def _detect_policy_category(query: str) -> str | None:
    """Auto-detect policy category from query keywords for targeted search."""
    query_lower = query.lower()
    
    category_keywords = {
        "Leave Policy": ["leave", "vacation", "holiday", "time off", "pto", "annual leave", "sick leave", "casual leave"],
        "Insurance & Benefits": ["insurance", "benefit", "medical", "health", "coverage", "claim", "dental", "vision"],
        "Maternity & Paternity": ["maternity", "paternity", "parental", "baby", "child", "pregnancy", "adoption"],
        "Code of Conduct": ["conduct", "ethics", "behavior", "harassment", "discrimination", "compliance"],
        "Compensation": ["salary", "compensation", "bonus", "incentive", "pay", "raise", "hike", "ctc"],
        "Performance": ["performance", "appraisal", "review", "rating", "kpi", "goal", "target"],
        "Training & Development": ["training", "learning", "development", "course", "certification", "skill"],
        "Travel & Expenses": ["travel", "expense", "reimbursement", "conveyance", "allowance"],
        "Work From Home": ["work from home", "wfh", "remote", "telecommute", "hybrid work"],
    }
    
    for category, keywords in category_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return category
    return None


def _highlight_companies(text: str, companies: list[str]) -> str:
    """Highlight company names with bold and add line breaks before new points."""
    if not companies or not text:
        return text

    import re

    # Sort by length descending to match longer names first
    sorted_companies = sorted(companies, key=len, reverse=True)

    # Build regex pattern for company names
    escaped = [re.escape(c) for c in sorted_companies]
    company_pattern = re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)

    # Remove "For [Company] ," prefix patterns
    text = re.sub(r'(?i)\bFor\s+(' + '|'.join(escaped) + r')\s*,\s*', r'\1: ', text)

    # Add line breaks at sentence boundaries before key points
    # Split on period followed by space and a capital letter (new sentence)
    text = re.sub(r'\.\s+(?=[A-Z][a-z])', '.\n\n', text)

    return text


def _apply_company_highlight_html(text: str, companies: list[str]) -> str:
    """Apply HTML highlighting to company names in already-escaped text."""
    if not companies or not text:
        return text

    import re

    sorted_companies = sorted(companies, key=len, reverse=True)
    escaped = [re.escape(c) for c in sorted_companies]
    company_pattern = re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)

    # Replace company names with styled span
    text = company_pattern.sub(r'<span class="company-highlight">\1</span>', text)

    # Convert newlines to <br> for HTML rendering
    text = text.replace("\n", "<br>")

    return text


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # New Chat at top
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">🆕 New Chat</div>', unsafe_allow_html=True)
    if st.button("➕ Start New Conversation", key="new_chat_btn", use_container_width=True):
        st.session_state.last_result = None
        st.success("Started a new conversation.")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Online / Offline Mode Toggle
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">🌐 Connection Mode</div>', unsafe_allow_html=True)
    online_mode = st.toggle(
        "Online Mode (Cloud API)",
        value=True,
        key="online_mode_toggle",
        help="ON = Cloud API (HuggingFace). OFF = Local model (no internet needed).",
    )
    if online_mode:
        st.markdown('<span class="status-badge status-online">● Online — Cloud API</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-offline">● Offline — Local Model</span>', unsafe_allow_html=True)
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

    if online_mode:
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
    else:
        selected_model = "Qwen/Qwen2.5-0.5B-Instruct"
        st.markdown(
            "<div style='font-size:0.85rem;color:#10b981;margin-top:0.25rem;'>"
            "🚀 <b>Qwen2.5-0.5B</b> — Fastest local model (~1GB download once)"
            "</div>",
            unsafe_allow_html=True,
        )

    rag_mode = st.radio(
        "RAG Mode",
        ["fast", "hybrid", "semantic", "fullylexical"],
        format_func=lambda x: {
            "fast":         "⚡ Fast (No Reranking)",
            "hybrid":       "🔀 Hybrid (Recommended)",
            "semantic":     "🔵 Semantic Only",
            "fullylexical": "🔤 Lexical Only",
        }[x],
        key="rag_mode_radio",
    )

    use_local = not online_mode
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
    
    # Policy category / subfolder selector
    _policy_categories = [
        "Leave Policy", "Insurance & Benefits", "Maternity & Paternity",
        "Code of Conduct", "Compensation", "Performance", "Training & Development",
        "Travel & Expenses", "Work From Home", "General"
    ]
    upload_subfolder = st.selectbox(
        "Policy Category", _policy_categories,
        index=len(_policy_categories) - 1, key="upload_subfolder_select",
        help="Categorize the policy for organized retrieval"
    )
    
    uploaded = st.file_uploader(
        "Upload HR Policies", type=["pdf", "txt", "docx"],
        accept_multiple_files=True, key="doc_uploader", label_visibility="collapsed",
    )
    if uploaded and st.button("📥 Index Documents", key="index_btn", use_container_width=True):
        with st.spinner("Indexing documents to Firestore + FAISS..."):
            try:
                with Timer("Total upload") as t_upload:
                    retriever = get_retriever()
                    tenant = upload_company
                    
                    # Parse files in-memory with subfolder metadata
                    with Timer("Parse files") as t_parse:
                        from src.data_loader import parse_uploaded_files
                        all_docs = parse_uploaded_files(uploaded, company=tenant, subfolder=upload_subfolder)
                    
                    if all_docs:
                        # Send to Firestore via embedding pipeline
                        with Timer("Add to Firestore + BM25") as t_fs:
                            retriever.pipeline.add_documents(all_docs)
                            st.session_state.docs_indexed += len(all_docs)
                        
                        # Also add to FAISS index for fast search
                        if retriever.faiss_index:
                            try:
                                with Timer("Add to FAISS") as t_faiss:
                                    # Get the chunks that were just added
                                    chunks = retriever.pipeline._bm25_docs[-len(all_docs):]
                                    vectors = []
                                    doc_ids = []
                                    metadata_list = []
                                    for chunk in chunks:
                                        # Compute embedding for new chunk
                                        emb = retriever.pipeline._store._call_hf_cloud_embedding(chunk.get("chunk_text", ""))
                                        if emb and len(emb) == 1024:
                                            vectors.append(emb)
                                            doc_ids.append(chunk.get("doc_id", ""))
                                            metadata_list.append(chunk.get("metadata", {}))
                                    if vectors:
                                        retriever.faiss_index.add_vectors(vectors, doc_ids, metadata_list)
                                        print(f"[App] Added {len(vectors)} vectors to FAISS index")
                            except Exception as exc:
                                print(f"[App] FAISS update notice: {exc}")
                        
                        store = retriever.pipeline.vector_store
                        if store.is_online:
                            st.success(
                                f"Indexed {len(all_docs)} chunks for '{tenant}' — "
                                f"sent to Firestore + FAISS."
                            )
                        else:
                            st.warning(
                                f"Indexed {len(all_docs)} chunks for '{tenant}'. "
                                f"Firebase is offline — data stored in memory only."
                            )
                    else:
                        st.warning("No processable content found in uploaded files.")
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
        st.session_state.last_result = None
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
tab_search, tab_docs = st.tabs(["🔍 Search", "📚 Documents"])
st.markdown('</div>', unsafe_allow_html=True)


# ── Search tab (Chat Interface) ────────────────────────────────────────────────
with tab_search:
    # Render chat history
    for turn in st.session_state.chat_history:
        # User message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(turn["query"])

        # Assistant message
        result = turn["result"]
        gen = result.get("generation", {})
        answers = gen.get("answers", [])
        is_multi_company = gen.get("multi_company", False)

        with st.chat_message("assistant", avatar="🤖"):
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
                    raw_text = ans.get("answer", "")
                    text = _highlight_companies(raw_text, companies)
                    rgb = _hex_to_rgb(bc)
                    card_class = {
                        "general": "multi-company-general",
                        "differences": "multi-company-diff",
                        "summary": "multi-company-summary",
                    }.get(ans.get("section_type", "general"), "answer-card")
                    import html as html_mod
                    safe_text = html_mod.escape(text)
                    safe_text = _apply_company_highlight_html(safe_text, companies)
                    st.markdown(f"""
<div class="{card_class}">
    <div class="answer-card-header">
        <span style="font-size:1.2rem;">{icon}</span>
        <span class="provider-badge" style="background:rgba({rgb},0.15);color:{bc};border:1px solid {bc}40;">{label}</span>
    </div>
    <div class="answer-text">{safe_text}</div>
</div>""", unsafe_allow_html=True)
            else:
                _all_companies = []
                if data_path.exists():
                    _all_companies = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
                for ans in answers:
                    bc = ans.get("badge_color", "#6366f1")
                    icon = ans.get("icon", "🤖")
                    label = ans.get("label", "Policy Processor")
                    raw_text = ans.get("answer", "")
                    text = _highlight_companies(raw_text, _all_companies)
                    rgb = _hex_to_rgb(bc)
                    import html as html_mod
                    safe_text = html_mod.escape(text)
                    safe_text = _apply_company_highlight_html(safe_text, _all_companies)
                    st.markdown(f"""
<div class="answer-card">
    <div class="answer-card-header">
        <span style="font-size:1.2rem;">{icon}</span>
        <span class="provider-badge" style="background:rgba({rgb},0.15);color:{bc};border:1px solid {bc}40;">{label}</span>
    </div>
    <div class="answer-text">{safe_text}</div>
</div>""", unsafe_allow_html=True)

        if result.get("mode") in ("bm25_only", "fast"):
            st.markdown(
                '<span class="status-badge status-ok">'
                "✅ Search complete"
                "</span>",
                unsafe_allow_html=True,
            )

        # Detailed retrieval results
        with st.expander("🔍 Detailed Results", expanded=False):
            detail_r, detail_l, detail_s = st.tabs(["🔀 Reranked", "🔤 Lexical", "🔵 Semantic"])
            render_chunks(detail_r, result.get("reranked", []))
            render_chunks(detail_l, result.get("lexical", []))
            render_chunks(detail_s, result.get("semantic", []))


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
        with st.spinner("🔍 Searching documents..."):
            try:
                with Timer("Total query") as t_total:
                    retriever = get_retriever()
                    t_prep = time.perf_counter()
                    company_arg = None if selected_company == "All Companies" else selected_company
                    
                    # Auto-detect policy category from query for targeted search
                    detected_subfolder = _detect_policy_category(user_query)
                    if detected_subfolder:
                        print(f"[App] Detected policy category: {detected_subfolder}")
                    print(f"[Timer] Prep: {time.perf_counter() - t_prep:.3f}s")
                    
                    # Fast mode: reduce search scope for speed
                    if rag_mode == "fast":
                        fast_top_k = min(top_k, 2)
                        fast_rerank = min(rerank_top_n, 2)
                        fast_semantic = 5
                    else:
                        fast_top_k = min(top_k, 3)
                        fast_rerank = min(rerank_top_n, 3)
                        fast_semantic = 10
                    
                    result = retriever.search(
                        query=user_query,
                        top_k=fast_top_k,
                        company=company_arg,
                        subfolder=detected_subfolder,
                        rerank_top_n=fast_rerank,
                        semantic_limit=fast_semantic,
                        rag_mode=rag_mode,
                        hf_model=selected_model,
                        use_local_engine=use_local,
                    )
                st.session_state.last_result = result
                st.session_state.chat_history.append({"query": user_query, "result": result})
                st.rerun()
            except Exception as exc:
                st.error(f"Search error: {exc}")

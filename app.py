"""
app.py – LOVA_HR | Firebase-Powered Hybrid RAG Engine
BM25 Lexical + Firestore Semantic (BGE-M3) + RRF Fusion + BGE Reranker V2 M3.
Organized by company and subfolders. No generative LLM — 100% retrieval-based.
"""
import streamlit as st
import os

# ── Environment & Performance Optimization Flags ───────────────────────────
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil
import stat
import hashlib
import html
import re
from pathlib import Path
from typing import List, Dict, Any


def _safe_rmtree(path: Path, retries: int = 3, delay: float = 0.5) -> None:
    """Remove a directory tree safely on Windows.

    Handles two common Windows failure modes:
    1. Read-only files (OneDrive sync, Windows marking uploads as RO):
       The onerror callback clears the read-only bit and retries.
    2. Transient file locks (Streamlit reruns, ChromaDB flushing):
       Retries up to `retries` times with `delay` seconds between attempts.
    """
    import time

    def _handle_readonly(func, fpath, exc_info):
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception as e:
            print(f"[App] Could not remove '{fpath}': {e}")

    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(str(path), onerror=_handle_readonly)
            return  # success
        except PermissionError as e:
            if attempt < retries:
                print(f"[App] rmtree attempt {attempt} failed ({e}) — retrying in {delay}s…")
                time.sleep(delay)
            else:
                print(f"[App] _safe_rmtree permanently failed for '{path}': {e}")
        except Exception as e:
            print(f"[App] _safe_rmtree unexpected error for '{path}': {e}")
            return

from src.embeddings import EmbeddingPipeline
from src.retriever import HybridRetriever
from src.config import is_firebase_available
from src.data_loader import load_all_documents

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LOVA_HR — Scoped RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hide default Streamlit chrome */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Custom Hero and Titles */
    .hero-container {
        text-align: center;
        padding: 1.2rem 0 0.8rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.4rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Scope indicator banner */
    .scope-banner {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 1.5rem;
        font-size: 0.88rem;
        color: #e2e8f0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .scope-label {
        font-weight: 700;
        color: #a5b4fc;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }

    /* Result cards */
    .result-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #6366f1;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .lexical-card {
        border-left-color: #ec4899; /* Pink accent */
    }
    .semantic-card {
        border-left-color: #6366f1; /* Indigo accent */
    }
    .reranked-card {
        border-left-color: #f59e0b; /* Amber accent — top quality results */
        background: rgba(245, 158, 11, 0.04);
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
        font-size: 0.8rem;
    }
    .doc-badge {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #cbd5e1;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.75rem;
    }
    .score-badge {
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    .lexical-score {
        color: #fbcfe8;
        background: rgba(236, 72, 153, 0.15);
    }
    .semantic-score {
        color: #e0e7ff;
        background: rgba(99, 102, 241, 0.15);
    }
    .reranked-score {
        color: #fef3c7;
        background: rgba(245, 158, 11, 0.20);
    }
    .card-body {
        color: #e2e8f0;
        font-size: 0.88rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    /* Firebase status badge */
    .firebase-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        letter-spacing: 0.03em;
    }
    .firebase-online {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #86efac;
    }
    .firebase-offline {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #fca5a5;
    }

    /* Sidebar Styling */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .library-section {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        padding: 0.6rem;
        margin-bottom: 0.8rem;
    }
    .company-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.2rem;
    }
    .subfolder-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        margin-left: 0.4rem;
        margin-top: 0.3rem;
        margin-bottom: 0.2rem;
    }
    .file-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.3rem 0.5rem;
        background: rgba(255, 255, 255, 0.01);
        border-radius: 4px;
        margin-left: 0.8rem;
        margin-bottom: 0.25rem;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }
    .file-info {
        font-size: 0.78rem;
        color: #cbd5e1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 80%;
    }
    .file-size {
        font-size: 0.68rem;
        color: #64748b;
        margin-left: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helper to render single-tab output and references ───────────────────────
def render_single_view_results(
    reranked_results: List[Dict[str, Any]],
    lexical_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    generation: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render output and reference information in a single unified container without multiple tabs.
    Shows grounded answer/snippet along with source metadata (company, subfolder, filename, score).
    """
    if generation and generation.get("answer"):
        provider = generation.get("provider", "LLM")
        ans_text = generation.get("answer", "")
        st.markdown(
            f'<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95)); border: 1px solid #38bdf8; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 4px 20px rgba(56, 189, 248, 0.15);">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">'
            f'<span style="font-weight:700; color:#38bdf8; font-size:1.05rem;">🤖 LOVA_HR AI Policy Answer</span>'
            f'<span style="background:rgba(56,189,248,0.15); color:#38bdf8; padding:0.25rem 0.6rem; border-radius:20px; font-size:0.75rem; font-weight:600;">{provider}</span>'
            f'</div>'
            f'<div style="color:#e2e8f0; font-size:0.95rem; line-height:1.6;">{ans_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    results = reranked_results if reranked_results else (semantic_results if semantic_results else lexical_results)

    if not results:
        st.info("No matching document context found for this query.")
        return

    st.markdown("### 📄 Grounded References & Source Chunks")
    for i, res in enumerate(results, 1):
        escaped_text = html.escape(res.get("chunk_text", ""))
        comp = html.escape(res.get("company", "General"))
        sub = html.escape(res.get("subfolder", "General"))
        fname = html.escape(res.get("filename", "document"))
        src = html.escape(res.get("source_file", "—"))

        r_score = res.get("reranker_score")
        if r_score is not None:
            score_badge = f"Reranker: {float(r_score):.4f}"
        elif "rrf_score" in res:
            score_badge = f"RRF: {float(res['rrf_score']):.4f}"
        elif "score" in res:
            score_badge = f"Score: {float(res['score']):.2f}"
        else:
            score_badge = "Relevance: Match"

        st.markdown(
            f'<div class="result-card reranked-card">'
            f'<div class="card-header">'
            f'<span class="doc-badge">#{i} | 🏢 {comp} &gt; 📁 {sub} &gt; 📄 {fname}</span>'
            f'<span class="score-badge reranked-score">{score_badge}</span>'
            f'</div>'
            f'<div class="card-body">{escaped_text}</div>'
            f'<div style="font-size:0.75rem; color:#94a3b8; margin-top:0.6rem; border-top:1px solid rgba(255,255,255,0.08); padding-top:0.4rem;">'
            f'<strong>Reference Source:</strong> {src}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ── Document Setup & State ───────────────────────────────────────────────────
DATA_DIR = Path("data")
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Session State Variables
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache pipeline and retriever (model loading is expensive)
@st.cache_resource(show_spinner="Loading HR Search Engine & Models...")
def get_pipeline() -> EmbeddingPipeline:
    return EmbeddingPipeline()

@st.cache_resource(show_spinner="Loading HR Search Engine & Models...")
def get_retriever() -> HybridRetriever:
    return HybridRetriever(pipeline=get_pipeline())

pipeline = get_pipeline()
retriever = get_retriever()

# Firebase connection status (checked once per session)
_firebase_status = is_firebase_available()

# Helpers
def get_existing_companies(data_dir: Path = DATA_DIR) -> List[str]:
    """List all company folder names under data/."""
    if not data_dir.exists():
        return []
    return sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and d.name not in ["chroma_db", "chroma_test"]
    ])

def get_company_subfolders(company: str, data_dir: Path = DATA_DIR) -> List[str]:
    """List all subfolder names under a company's directory.

    Always includes 'General' so callers always get at least one entry,
    even when no subdirectories exist on disk.
    """
    if not company or company == "All Companies":
        return ["General"]
    comp_dir = data_dir / company
    if not comp_dir.exists() or not comp_dir.is_dir():
        return ["General"]
    try:
        subdirs = sorted({
            p.relative_to(comp_dir).as_posix()
            for p in comp_dir.glob("**/*")
            if p.is_dir()
        })
    except Exception:
        subdirs = []
    # Always surface 'General' first, then any real subdirectories
    result = ["General"] + [s for s in subdirs if s != "General"]
    return result

def get_all_indexed_files(data_dir: Path = DATA_DIR) -> List[Path]:
    """Gather all supported document files organized by company/subfolders."""
    extensions = ["*.pdf", "*.txt", "*.docx"]
    files = []
    companies = get_existing_companies(data_dir)
    for company in companies:
        comp_path = data_dir / company
        for ext in extensions:
            files.extend(comp_path.glob(f"**/{ext}"))
    return sorted(files)

def format_size(size_in_bytes: int) -> str:
    """Format file size in human-readable unit."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.1f} KB"
    else:
        return f"{size_in_bytes / (1024 * 1024):.1f} MB"

def get_document_state_hash() -> str:
    """Compute a md5 hash representing the current set of documents (names, sizes, mtimes)."""
    files = get_all_indexed_files()
    sig = []
    for f in files:
        try:
            stat = f.stat()
            # Include relative path from DATA_DIR to uniquely represent file state
            rel = f.relative_to(DATA_DIR)
            sig.append(f"{rel}:{stat.st_size}:{stat.st_mtime}")
        except Exception:
            pass
    return hashlib.md5("|".join(sig).encode("utf-8")).hexdigest()

@st.cache_data(show_spinner=False)
def load_and_chunk_cached(doc_hash: str) -> List[Any]:
    """Cache loaded & split document chunks in memory by doc_hash."""
    docs = load_all_documents(str(DATA_DIR))
    return pipeline.chunk_documents(docs)

def reindex_knowledge_base(force_clear: bool = False):
    """Load, chunk, and index the current set of documents into Firestore."""
    if force_clear:
        pipeline.vector_store.clear()
        st.cache_data.clear()

    files = get_all_indexed_files()
    current_hash = get_document_state_hash()
    if not files:
        st.session_state.chunks = []
        st.session_state.doc_hash = current_hash
        return

    with st.spinner("⏳ Updating search indexes (Firestore + BM25)..."):
        # Fast cached chunk retrieval (instant if doc_hash is unchanged)
        chunks = load_and_chunk_cached(current_hash)
        # Sync to Firestore (idempotent, skips existing chunks)
        pipeline.sync_vector_store(chunks)
        # Invalidate BM25 cache so new docs are picked up
        pipeline.invalidate_bm25_cache()

        # Save to session state
        st.session_state.chunks = chunks
        st.session_state.doc_hash = current_hash

# Legacy Folder Migration
def migrate_legacy_folders():
    """Migrate legacy folders to company-based organization on startup."""
    legacy_pdf = DATA_DIR / "pdf_files"
    migrated = False

    if legacy_pdf.exists() and legacy_pdf.is_dir():
        # 1. TCS policies
        tcs_file = legacy_pdf / "tcs_hr_policies.pdf"
        if tcs_file.exists():
            dest = DATA_DIR / "TCS" / "Policies"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tcs_file), str(dest / tcs_file.name))
            migrated = True

        # 2. HCLTech blueprint
        hcl_file = legacy_pdf / "hcltech_extended_hr_blueprint.pdf"
        if hcl_file.exists():
            dest = DATA_DIR / "HCLTech" / "Blueprint"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(hcl_file), str(dest / hcl_file.name))
            migrated = True

        # 3. Code of conduct
        coc_file = legacy_pdf / "codeofconduct.pdf"
        if coc_file.exists():
            dest = DATA_DIR / "General" / "CodeOfConduct"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(coc_file), str(dest / coc_file.name))
            migrated = True

        try:
            _safe_rmtree(legacy_pdf)
        except Exception:
            pass

    legacy_txt = DATA_DIR / "txt_files"
    if legacy_txt.exists() and legacy_txt.is_dir():
        try:
            shutil.rmtree(legacy_txt)
        except Exception:
            pass
        migrated = True

    if migrated:
        st.toast("Migrated legacy folders to Company layout!")
        reindex_knowledge_base(force_clear=True)

# Run legacy migration once on startup
if "migrated" not in st.session_state:
    migrate_legacy_folders()
    st.session_state.migrated = True

def sanitize_folder_name(name: str) -> str:
    """Remove invalid path characters for cross-platform folder safety."""
    cleaned = re.sub(r'[\\/:*?"<>|]', "", name).strip()
    return cleaned


# Auto-reindex on startup or file-system changes
current_hash = get_document_state_hash()
if "doc_hash" not in st.session_state or st.session_state.doc_hash != current_hash or (not st.session_state.chunks and get_all_indexed_files()):
    reindex_knowledge_base()

# ── Sidebar layout ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏢 LOVA_HR Settings</div>', unsafe_allow_html=True)
    
    # Sidebar Navigation Tabs
    sb_tab1, sb_tab2, sb_tab3 = st.tabs(["🔍 Scope", "📤 Upload", "⚙️ Manage"])

    # ──────────────────────────────────────────
    # TAB 1: SEARCH SCOPE CONTROLLER
    # ──────────────────────────────────────────
    with sb_tab1:
        st.markdown("**Active Search Scope**")
        
        companies = get_existing_companies()
        scope_companies = ["All Companies"] + companies
        
        if "search_company" in st.session_state and st.session_state.search_company not in scope_companies:
            st.session_state.search_company = "All Companies"
        
        search_company = st.selectbox(
            "Company Scope",
            scope_companies,
            index=0,
            key="search_company"
        )
        
        # Reset subfolder scope if company selection changes
        if "last_search_company" not in st.session_state:
            st.session_state.last_search_company = search_company

        if st.session_state.last_search_company != search_company:
            st.session_state.search_subfolder = "All Subfolders"
            st.session_state.last_search_company = search_company

        search_subfolder = "All Subfolders"
        if search_company != "All Companies":
            subfolders = get_company_subfolders(search_company)
            scope_subfolders = ["All Subfolders"] + subfolders
            
            if "search_subfolder" in st.session_state and st.session_state.search_subfolder not in scope_subfolders:
                st.session_state.search_subfolder = "All Subfolders"

            search_subfolder = st.selectbox(
                "Subfolder Scope",
                scope_subfolders,
                index=0,
                key="search_subfolder"
            )
        
        st.divider()
        st.markdown("**Search Settings**")
        top_k = st.slider(
            "Top Chunks to Retrieve", 1, 5, 3,
            help="Number of context passages to retrieve for Lexical and Semantic searches separately."
        )

    # ──────────────────────────────────────────
    # TAB 2: ADD DOCUMENTS (UPLOADER)
    # ──────────────────────────────────────────
    with sb_tab2:
        st.markdown("**Add Files to Library**")
        existing_comps = get_existing_companies()
        
        if not existing_comps:
            st.info("No companies exist yet. Create your first company below:")
            with st.form("quick_create_company", clear_on_submit=True):
                quick_comp_name = st.text_input("New Company Name")
                if st.form_submit_button("✨ Create Company", use_container_width=True, type="primary"):
                    clean_comp = sanitize_folder_name(quick_comp_name)
                    if clean_comp:
                        (DATA_DIR / clean_comp / "General").mkdir(parents=True, exist_ok=True)
                        st.toast(f"Created company '{clean_comp}'")
                        st.rerun()
                    else:
                        st.error("Please enter a valid company name.")
        else:
            company_options = existing_comps + ["+ Create New Company"]

            if "upload_company" in st.session_state and st.session_state.upload_company not in company_options:
                st.session_state.upload_company = company_options[0]

            upload_company = st.selectbox(
                "Select Destination Company",
                company_options,
                key="upload_company"
            )
            
            final_company = upload_company
            if upload_company == "+ Create New Company":
                raw_company_inline = st.text_input("New Company Name", value="", key="inline_new_comp")
                final_company = sanitize_folder_name(raw_company_inline)
            
            # Reset upload_subfolder if upload_company changes
            if "last_upload_company" not in st.session_state:
                st.session_state.last_upload_company = final_company
            if st.session_state.last_upload_company != final_company:
                if "upload_subfolder" in st.session_state:
                    del st.session_state["upload_subfolder"]
                st.session_state.last_upload_company = final_company

            subfolders = get_company_subfolders(final_company) if final_company else ["General"]
            options_subfolder = subfolders + ["+ Create New Subfolder"]

            if "upload_subfolder" in st.session_state and st.session_state.upload_subfolder not in options_subfolder:
                st.session_state.upload_subfolder = options_subfolder[0]

            upload_subfolder = st.selectbox(
                "Select Destination Subfolder",
                options_subfolder,
                key="upload_subfolder"
            )
            
            final_subfolder = upload_subfolder
            if upload_subfolder == "+ Create New Subfolder":
                raw_subfolder = st.text_input(
                    "New Subfolder Name",
                    value="",
                    key="new_subfolder_inline"
                )
                final_subfolder = sanitize_folder_name(raw_subfolder)
                
            uploaded_files = st.file_uploader(
                "Select document (.txt, .pdf, .docx)",
                type=["txt", "pdf", "docx"],
                accept_multiple_files=True,
                key="doc_uploader"
            )
            
            if st.button("📤 Upload and Index", use_container_width=True, type="primary"):
                if not final_company:
                    st.error("Please specify a valid company name!")
                elif not final_subfolder:
                    st.error("Please specify a valid subfolder name!")
                elif not uploaded_files:
                    st.error("No files selected!")
                else:
                    dest_dir = DATA_DIR / final_company / final_subfolder
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    with st.spinner(f"⚡ Processing {len(uploaded_files)} document(s)..."):
                        files_saved = False
                        for uf in uploaded_files:
                            target_file = dest_dir / uf.name
                            with open(target_file, "wb") as f:
                                f.write(uf.getbuffer())
                            files_saved = True
                            
                        if files_saved:
                            st.cache_data.clear()
                            reindex_knowledge_base()
                            st.toast(f"Successfully uploaded & indexed {len(uploaded_files)} file(s)!")
                            st.rerun()

    # ──────────────────────────────────────────
    # TAB 3: MANAGE COMPANIES & SUBFOLDERS
    # ──────────────────────────────────────────
    with sb_tab3:
        st.markdown("**Manage Folders**")
        
        # Form to Create Company
        with st.form("create_company_form", clear_on_submit=True):
            st.markdown("Create New Company")
            raw_comp = st.text_input("Company Name")
            if st.form_submit_button("✨ Create Company", use_container_width=True, type="primary"):
                new_comp = sanitize_folder_name(raw_comp)
                if new_comp:
                    # Create default general subfolder on local disk
                    (DATA_DIR / new_comp / "General").mkdir(parents=True, exist_ok=True)
                    st.cache_data.clear()
                    reindex_knowledge_base()
                    st.toast(f"Created company '{new_comp}' on local disk & Firebase!")
                    st.rerun()
                else:
                    st.error("Please enter a valid company name.")

        companies = get_existing_companies()
        if companies:
            st.divider()
            st.markdown("Company Operations")

            if "manage_comp" in st.session_state and st.session_state.manage_comp not in companies:
                st.session_state.manage_comp = companies[0]

            manage_comp = st.selectbox("Select Company", companies, key="manage_comp")
            
            # Form to Create Subfolder in Selected Company
            with st.form("create_subfolder_form", clear_on_submit=True):
                st.markdown(f"Create New Subfolder in '{manage_comp}'")
                raw_sub = st.text_input("Subfolder Name")
                if st.form_submit_button("✨ Create Subfolder", use_container_width=True):
                    new_sub = sanitize_folder_name(raw_sub)
                    if new_sub:
                        (DATA_DIR / manage_comp / new_sub).mkdir(parents=True, exist_ok=True)
                        st.cache_data.clear()
                        reindex_knowledge_base()
                        st.toast(f"Created subfolder '{new_sub}' in '{manage_comp}' on local disk & Firebase!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid subfolder name.")

            # Reset manage_sub if manage_comp changes
            if "last_manage_comp" not in st.session_state:
                st.session_state.last_manage_comp = manage_comp
            if st.session_state.last_manage_comp != manage_comp:
                if "manage_sub" in st.session_state:
                    del st.session_state["manage_sub"]
                st.session_state.last_manage_comp = manage_comp

            # Delete Company Button
            if st.button("🗑️ Delete Company", type="secondary", use_container_width=True):
                comp_dir = DATA_DIR / manage_comp
                # 1. Delete all vector chunks for this company from Firebase & memory
                pipeline.vector_store.delete_company_chunks(manage_comp)

                # 2. Delete company directory from local disk
                if comp_dir.exists():
                    _safe_rmtree(comp_dir)

                # 3. Invalidate BM25 cache
                pipeline.invalidate_bm25_cache()

                # 4. Clean session state keys for the deleted company
                for key in ["manage_comp", "upload_company", "search_company"]:
                    if key in st.session_state and st.session_state[key] == manage_comp:
                        del st.session_state[key]

                st.toast(f"Deleted company '{manage_comp}' from local disk & Firebase")
                st.cache_data.clear()
                reindex_knowledge_base()
                st.rerun()
                
            st.markdown("Subfolder Operations")
            subfolders = get_company_subfolders(manage_comp)  # always non-empty

            if "manage_sub" in st.session_state and st.session_state.manage_sub not in subfolders:
                st.session_state.manage_sub = subfolders[0]

            manage_sub = st.selectbox("Select Subfolder", subfolders, key="manage_sub")
            
            # Delete Subfolder Button
            if st.button("🗑️ Delete Subfolder", type="secondary", use_container_width=True):
                if manage_sub == "General":
                    st.error(
                        "'General' is the default subfolder and cannot be deleted. "
                        "Delete individual files instead, or delete the whole company."
                    )
                else:
                    sub_dir = DATA_DIR / manage_comp / manage_sub
                    # 1. Delete all vector chunks for this subfolder from Firebase & memory
                    pipeline.vector_store.delete_subfolder_chunks(manage_comp, manage_sub)

                    # 2. Delete subfolder directory from local disk
                    if sub_dir.exists():
                        _safe_rmtree(sub_dir)

                    # 3. Invalidate BM25 cache
                    pipeline.invalidate_bm25_cache()

                    # 4. Clean session state keys for the deleted subfolder
                    for key in ["manage_sub", "upload_subfolder", "search_subfolder"]:
                        if key in st.session_state and st.session_state[key] == manage_sub:
                            del st.session_state[key]

                    st.toast(f"Deleted subfolder '{manage_sub}' from '{manage_comp}' on local disk & Firebase")
                    st.cache_data.clear()
                    reindex_knowledge_base()
                    st.rerun()
                
    # ── Firebase Status Indicator ─────────────────────────────────────────────
    st.divider()
    if _firebase_status:
        st.markdown(
            '<span class="firebase-badge firebase-online">🔥 Firebase Connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="firebase-badge firebase-offline">⚠️ Firebase Offline (BM25 only)</span>',
            unsafe_allow_html=True,
        )
        st.caption("Place `serviceAccountKey.json` in repo root to enable semantic search.")

    # ── Document Library View (Sidebar Bottom) ────────────────────────────────
    st.divider()
    st.markdown("**Document Library**")
    
    files = get_all_indexed_files()
    if not files:
        st.info("No documents in the library.")
    else:
        # Group files by Company > Subfolder
        library_data: Dict[str, Dict[str, List[Path]]] = {}
        for f in files:
            # Structure is: data / company / subfolder / filename
            try:
                rel = f.relative_to(DATA_DIR)
                comp = rel.parts[0]
                sub = rel.parts[1]
                if comp not in library_data:
                    library_data[comp] = {}
                if sub not in library_data[comp]:
                    library_data[comp][sub] = []
                library_data[comp][sub].append(f)
            except Exception:
                pass
                
        # Render grouped files
        for comp, subs in library_data.items():
            with st.container(border=True):
                safe_comp = html.escape(comp)
                st.markdown(f'<div class="company-title">🏢 {safe_comp}</div>', unsafe_allow_html=True)
                for sub, paths in subs.items():
                    safe_sub = html.escape(sub)
                    st.markdown(f'<div class="subfolder-title">📁 {safe_sub}</div>', unsafe_allow_html=True)
                    for path in paths:
                        size_str = format_size(path.stat().st_size)
                        raw_name = path.name
                        short_name = raw_name if len(raw_name) < 18 else raw_name[:15] + "..."
                        safe_short_name = html.escape(short_name)
                        safe_full_name = html.escape(raw_name)
                        
                        col_info, col_del = st.columns([8, 2])
                        with col_info:
                            st.markdown(
                                f'<div class="file-item">'
                                f'<span class="file-info" title="{safe_full_name}">📄 {safe_short_name} <span class="file-size">({size_str})</span></span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        with col_del:
                            if st.button("❌", key=f"del_{comp}_{sub}_{path.name}", help=f"Delete {path.name}"):
                                try:
                                    # 1. Delete from Firestore using absolute path
                                    pipeline.vector_store.delete_file_chunks(str(path.resolve()))
                                    # 2. Invalidate BM25 cache
                                    pipeline.invalidate_bm25_cache()
                                    # 3. Delete file
                                    if path.exists():
                                        path.unlink()
                                    st.toast(f"Removed {raw_name}")
                                    reindex_knowledge_base()
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")

# ── Main Chat Interface ───────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">LOVA_HR ⚡</h1>
        <div class="hero-subtitle">Firebase · BGE-M3 · RRF Fusion · BGE Reranker · Zero Hallucinations</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display Active Search Scope Banner
scope_text = "All Companies"
if search_company != "All Companies":
    scope_text = f"🏢 {search_company}"
    if search_subfolder != "All Subfolders":
        scope_text += f" &nbsp;>&nbsp; 📁 {search_subfolder}"

st.markdown(
    f'<div class="scope-banner">'
    f'<span class="scope-label">Active Scope:</span>'
    f'<span>{scope_text}</span>'
    f'</div>',
    unsafe_allow_html=True
)

# Render Conversation History
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            if msg.get("content"):
                st.warning(msg["content"])
            if msg.get("error"):
                st.error(msg["error"])

            lexical_results  = msg.get("lexical_results", [])
            semantic_results = msg.get("semantic_results", [])
            reranked_results = msg.get("reranked_results", [])

            if lexical_results or semantic_results or reranked_results:
                render_single_view_results(
                    reranked_results=reranked_results,
                    lexical_results=lexical_results,
                    semantic_results=semantic_results,
                )

# Chat Input Handler
if query := st.chat_input("Ask a question about the documents in scope…"):
    
    # User turn
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    # Assistant turn
    with st.chat_message("assistant"):
        if not st.session_state.chunks and not _firebase_status:
            st.warning("Document library is empty. Please upload documents in the sidebar first.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Document library is empty. Please upload documents in the sidebar first.",
                "lexical_results": [],
                "semantic_results": [],
                "reranked_results": [],
                "error": None,
            })
        else:
            with st.spinner("🔍 Querying Firebase + BM25 + BGE Reranker…"):
                search_response = retriever.search(
                    query=query,
                    top_k=top_k,
                    company=search_company,
                    subfolder=search_subfolder,
                    rerank_top_n=5,
                )
                lexical_results  = search_response["lexical"]
                semantic_results = search_response["semantic"]
                reranked_results = search_response["reranked"]

            if search_response.get("error"):
                st.error(search_response["error"])
                
            render_single_view_results(
                reranked_results=reranked_results,
                lexical_results=lexical_results,
                semantic_results=semantic_results,
            )

            # Save response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": "",
                "lexical_results":  lexical_results,
                "semantic_results": semantic_results,
                "reranked_results": reranked_results,
                "error": search_response.get("error"),
            })

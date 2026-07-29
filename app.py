"""
app.py – LOVA_HR | Hybrid Lexical + Semantic Scoped RAG Engine
Pure retrieval mode — organized by company and subfolders. No external LLMs used.
"""
import streamlit as st
import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from src.embeddings import EmbeddingPipeline
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
    .card-body {
        color: #e2e8f0;
        font-size: 0.88rem;
        line-height: 1.6;
        white-space: pre-wrap;
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

# ── Document Setup & State ───────────────────────────────────────────────────
DATA_DIR = Path("data")
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Session State Variables
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache pipeline model loading
@st.cache_resource
def get_pipeline():
    return EmbeddingPipeline()

pipeline = get_pipeline()

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
    """List all subfolder names under a company's directory."""
    comp_dir = data_dir / company
    if not comp_dir.exists():
        return ["General"]
    subdirs = sorted([
        d.name for d in comp_dir.iterdir()
        if d.is_dir()
    ])
    if not subdirs:
        return ["General"]
    return subdirs

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

def reindex_knowledge_base(force_clear: bool = False):
    """Load, chunk, and index the current set of documents."""
    if force_clear:
        pipeline.vector_store.clear()

    files = get_all_indexed_files()
    if not files:
        st.session_state.chunks = []
        st.session_state.doc_hash = get_document_state_hash()
        return

    with st.spinner("⏳ Loading organized documents & updating search indexes..."):
        # Load documents
        docs = load_all_documents(str(DATA_DIR))
        # Chunk
        chunks = pipeline.chunk_documents(docs)
        # Sync Vector DB (idempotent, skips existing)
        pipeline.sync_vector_store(chunks)
        
        # Save to session state
        st.session_state.chunks = chunks
        st.session_state.doc_hash = get_document_state_hash()

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
            shutil.rmtree(legacy_pdf)
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

# Auto-reindex on startup or file-system changes
current_hash = get_document_state_hash()
if "doc_hash" not in st.session_state or st.session_state.doc_hash != current_hash:
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
        
        search_company = st.selectbox(
            "Company Scope",
            scope_companies,
            index=0,
            key="search_company"
        )
        
        search_subfolder = "All Subfolders"
        if search_company != "All Companies":
            subfolders = get_company_subfolders(search_company)
            scope_subfolders = ["All Subfolders"] + subfolders
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
        companies = get_existing_companies()
        
        if not companies:
            st.warning("Create a company in the 'Manage' tab first!")
        else:
            upload_company = st.selectbox(
                "Select Destination Company",
                companies,
                key="upload_company"
            )
            
            subfolders = get_company_subfolders(upload_company)
            options_subfolder = subfolders + ["+ Create New Subfolder"]
            upload_subfolder = st.selectbox(
                "Select Destination Subfolder",
                options_subfolder,
                key="upload_subfolder"
            )
            
            final_subfolder = upload_subfolder
            if upload_subfolder == "+ Create New Subfolder":
                final_subfolder = st.text_input(
                    "New Subfolder Name",
                    value="",
                    key="new_subfolder_inline"
                ).strip()
                
            uploaded_files = st.file_uploader(
                "Select document (.txt, .pdf, .docx)",
                type=["txt", "pdf", "docx"],
                accept_multiple_files=True,
                key="doc_uploader"
            )
            
            if st.button("📤 Upload and Index", use_container_width=True, type="primary"):
                if not final_subfolder:
                    st.error("Please specify a subfolder name!")
                elif not uploaded_files:
                    st.error("No files selected!")
                else:
                    dest_dir = DATA_DIR / upload_company / final_subfolder
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    files_saved = False
                    for uf in uploaded_files:
                        target_file = dest_dir / uf.name
                        with open(target_file, "wb") as f:
                            f.write(uf.getbuffer())
                        files_saved = True
                        
                    if files_saved:
                        st.success("Uploaded successfully!")
                        reindex_knowledge_base()
                        st.rerun()

    # ──────────────────────────────────────────
    # TAB 3: MANAGE COMPANIES & SUBFOLDERS
    # ──────────────────────────────────────────
    with sb_tab3:
        st.markdown("**Manage Folders**")
        
        # Form to Create Company
        with st.form("create_company_form", clear_on_submit=True):
            st.markdown("Create New Company")
            new_comp = st.text_input("Company Name").strip()
            if st.form_submit_button("Create Company", use_container_width=True):
                if new_comp:
                    # Create default general subfolder to create the path
                    (DATA_DIR / new_comp / "General").mkdir(parents=True, exist_ok=True)
                    st.success(f"Company '{new_comp}' created!")
                    st.rerun()
                else:
                    st.error("Please enter a valid company name.")
                    
        # Rename or Delete Subfolders
        companies = get_existing_companies()
        if companies:
            st.divider()
            st.markdown("Company Operations")
            manage_comp = st.selectbox("Select Company", companies, key="manage_comp")
            
            # Delete Company Button
            if st.button("🗑️ Delete Company", type="secondary", use_container_width=True):
                comp_dir = DATA_DIR / manage_comp
                # 1. Delete all chunks of files in this company
                comp_files = get_all_indexed_files(DATA_DIR)
                for f in comp_files:
                    if manage_comp in f.parts:
                        pipeline.vector_store.delete_file_chunks(str(f.resolve()))
                
                # 2. Delete company directory
                if comp_dir.exists():
                    shutil.rmtree(comp_dir)
                    
                st.toast(f"Deleted company '{manage_comp}'")
                reindex_knowledge_base()
                st.rerun()
                
            st.markdown("Subfolder Operations")
            subfolders = get_company_subfolders(manage_comp)
            manage_sub = st.selectbox("Select Subfolder", subfolders, key="manage_sub")
            
            # Delete Subfolder Button
            if st.button("🗑️ Delete Subfolder", type="secondary", use_container_width=True):
                sub_dir = DATA_DIR / manage_comp / manage_sub
                # 1. Delete all chunks
                comp_files = get_all_indexed_files(DATA_DIR)
                for f in comp_files:
                    if manage_comp in f.parts and manage_sub in f.parts:
                        pipeline.vector_store.delete_file_chunks(str(f.resolve()))
                        
                # 2. Delete subfolder
                if sub_dir.exists():
                    shutil.rmtree(sub_dir)
                    
                st.toast(f"Deleted subfolder '{manage_sub}' from '{manage_comp}'")
                reindex_knowledge_base()
                st.rerun()
                
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
                st.markdown(f'<div class="company-title">🏢 {comp}</div>', unsafe_allow_html=True)
                for sub, paths in subs.items():
                    st.markdown(f'<div class="subfolder-title">📁 {sub}</div>', unsafe_allow_html=True)
                    for path in paths:
                        size_str = format_size(path.stat().st_size)
                        short_name = path.name if len(path.name) < 18 else path.name[:15] + "..."
                        
                        col_info, col_del = st.columns([8, 2])
                        with col_info:
                            st.markdown(
                                f'<div class="file-item">'
                                f'<span class="file-info" title="{path.name}">📄 {short_name} <span class="file-size">({size_str})</span></span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        with col_del:
                            if st.button("❌", key=f"del_{comp}_{sub}_{path.name}", help=f"Delete {path.name}"):
                                try:
                                    # 1. Delete from vector DB using absolute path
                                    pipeline.vector_store.delete_file_chunks(str(path.resolve()))
                                    # 2. Delete file
                                    if path.exists():
                                        path.unlink()
                                    st.toast(f"Removed {path.name}")
                                    reindex_knowledge_base()
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")

# ── Main Chat Interface ───────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">LOVA_HR ⚡</h1>
        <div class="hero-subtitle">Zero-LLM Company-Scoped RAG Engine</div>
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
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            lexical_results = msg.get("lexical_results", [])
            semantic_results = msg.get("semantic_results", [])
            
            tab_lex, tab_sem = st.tabs(["🔍 Lexical Retrieval (BM25)", "🧠 Semantic Retrieval (Dense Vector)"])
            
            with tab_lex:
                if not lexical_results:
                    st.info("No matching lexical context found.")
                else:
                    for i, res in enumerate(lexical_results, 1):
                        st.markdown(
                            f'<div class="result-card lexical-card">'
                            f'<div class="card-header">'
                            f'<span class="doc-badge">#{i} | {res["company"]} > {res["subfolder"]} > {res["filename"]}</span>'
                            f'<span class="score-badge lexical-score">BM25: {res["score"]:.2f}</span>'
                            f'</div>'
                            f'<div class="card-body">{res["chunk_text"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
            with tab_sem:
                if not semantic_results:
                    st.info("No matching semantic context found.")
                else:
                    for i, res in enumerate(semantic_results, 1):
                        st.markdown(
                            f'<div class="result-card semantic-card">'
                            f'<div class="card-header">'
                            f'<span class="doc-badge">#{i} | {res["company"]} > {res["subfolder"]} > {res["filename"]}</span>'
                            f'<span class="score-badge semantic-score">Similarity: {res["score"]*100:.1f}%</span>'
                            f'</div>'
                            f'<div class="card-body">{res["chunk_text"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

# Chat Input Handler
if query := st.chat_input("Ask a question about the documents in scope…"):
    
    # User turn
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    # Assistant turn
    with st.chat_message("assistant"):
        if not st.session_state.chunks:
            st.warning("Document library is empty. Please upload documents in the sidebar first.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "",
                "lexical_results": [],
                "semantic_results": []
            })
        else:
            with st.spinner("🔍 Querying scoped database…"):
                # Scoped Lexical Search
                lexical_results = pipeline.lexical_search(
                    query=query,
                    chunks=st.session_state.chunks,
                    top_k=top_k,
                    company=search_company,
                    subfolder=search_subfolder
                )
                
                # Scoped Semantic Search
                semantic_results = pipeline.semantic_search(
                    query=query,
                    top_k=top_k,
                    company=search_company,
                    subfolder=search_subfolder
                )
                
            tab_lex, tab_sem = st.tabs(["🔍 Lexical Retrieval (BM25)", "🧠 Semantic Retrieval (Dense Vector)"])
            
            with tab_lex:
                if not lexical_results:
                    st.info("No matching lexical context found.")
                else:
                    for i, res in enumerate(lexical_results, 1):
                        st.markdown(
                            f'<div class="result-card lexical-card">'
                            f'<div class="card-header">'
                            f'<span class="doc-badge">#{i} | {res["company"]} > {res["subfolder"]} > {res["filename"]}</span>'
                            f'<span class="score-badge lexical-score">BM25: {res["score"]:.2f}</span>'
                            f'</div>'
                            f'<div class="card-body">{res["chunk_text"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
            with tab_sem:
                if not semantic_results:
                    st.info("No matching semantic context found.")
                else:
                    for i, res in enumerate(semantic_results, 1):
                        st.markdown(
                            f'<div class="result-card semantic-card">'
                            f'<div class="card-header">'
                            f'<span class="doc-badge">#{i} | {res["company"]} > {res["subfolder"]} > {res["filename"]}</span>'
                            f'<span class="score-badge semantic-score">Similarity: {res["score"]*100:.1f}%</span>'
                            f'</div>'
                            f'<div class="card-body">{res["chunk_text"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            
            # Save response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": "",
                "lexical_results": lexical_results,
                "semantic_results": semantic_results
            })
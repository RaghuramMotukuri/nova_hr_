"""
data_loader.py
Load documents (PDF, TXT, DOCX) from the data directory,
extract company/subfolder metadata from their paths,
and return a flat list of LangChain Document objects.
"""
from pathlib import Path
from typing import List, Any, Dict


def extract_metadata_from_path(fpath: Path, data_dir: Path) -> Dict[str, str]:
    """
    Extract company, subfolder, and filename from the file path.
    Assumes structure: data_dir / company_name / subfolder_name / filename
    """
    try:
        relative_path = fpath.resolve().relative_to(data_dir.resolve())
        parts = relative_path.parts
        if len(parts) >= 3:
            company = parts[0]
            # Subfolder is everything in between company and filename, joined by '/'
            subfolder = "/".join(parts[1:-1])
            filename = parts[-1]
        elif len(parts) == 2:
            company = parts[0]
            subfolder = "General"
            filename = parts[-1]
        else:
            company = "General"
            subfolder = "General"
            filename = fpath.name
    except Exception:
        company = "General"
        subfolder = "General"
        filename = fpath.name
        
    return {
        "company": company,
        "subfolder": subfolder,
        "filename": filename,
        "source": str(fpath.resolve()),
    }


def load_all_documents(data_dir: str = "data") -> List[Any]:
    """
    Recursively load all supported files from *data_dir*.
    Supported formats: PDF, TXT, DOCX.

    Returns a list of LangChain Document objects (each enriched with
    company, subfolder, filename and source in metadata).
    """
    from langchain_core.documents import Document

    data_path = Path(data_dir).resolve()

    if not data_path.exists():
        print(f"[DataLoader] '{data_path}' not found - creating empty directory.")
        data_path.mkdir(parents=True, exist_ok=True)
        return []

    documents: List[Any] = []

    # ── PDFs ─────────────────────────────────────────────────────────────────
    pdf_files = [f for f in data_path.glob("**/*.pdf") if not f.name.startswith("~$") and not f.name.startswith(".")]
    print(f"[DataLoader] **/*.pdf : found {len(pdf_files)} file(s)")
    if pdf_files:
        try:
            import fitz  # PyMuPDF
            has_fitz = True
        except ImportError:
            has_fitz = False

        for fpath in pdf_files:
            try:
                meta = extract_metadata_from_path(fpath, data_path)
                if has_fitz:
                    try:
                        # Context manager guarantees the file handle is released
                        # even if an exception is raised mid-loop (prevents WinError 5
                        # when the folder is later deleted with shutil.rmtree).
                        with fitz.open(str(fpath)) as doc_pdf:
                            for page_num, page in enumerate(doc_pdf):
                                text = page.get_text()
                                if text and text.strip():
                                    page_meta = meta.copy()
                                    page_meta["page"] = page_num + 1
                                    documents.append(Document(page_content=text, metadata=page_meta))
                    except Exception as fitz_exc:
                        print(f"[DataLoader] PyMuPDF notice for '{fpath.name}': {fitz_exc} - using PyPDFLoader fallback")
                        from langchain_community.document_loaders import PyPDFLoader
                        docs = PyPDFLoader(str(fpath)).load()
                        for d in docs:
                            d.metadata.update(meta)
                        documents.extend(docs)
                else:
                    from langchain_community.document_loaders import PyPDFLoader
                    docs = PyPDFLoader(str(fpath)).load()
                    for d in docs:
                        d.metadata.update(meta)
                    documents.extend(docs)
            except Exception as exc:
                print(f"[DataLoader] [Warning] Skipping '{fpath.name}': {exc}")

    # ── TXT files ─────────────────────────────────────────────────────────────
    txt_files = [f for f in data_path.glob("**/*.txt") if not f.name.startswith("~$") and not f.name.startswith(".")]
    print(f"[DataLoader] **/*.txt  : found {len(txt_files)} file(s)")
    for fpath in txt_files:
        try:
            try:
                text = fpath.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    text = fpath.read_text(encoding="latin-1")
                except Exception:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                    
            if text.strip():
                meta = extract_metadata_from_path(fpath, data_path)
                documents.append(Document(page_content=text, metadata=meta))
        except Exception as exc:
            print(f"[DataLoader] [Warning] Skipping '{fpath.name}': {exc}")

    # ── DOCX files ────────────────────────────────────────────────────────────
    docx_files = [f for f in data_path.glob("**/*.docx") if not f.name.startswith("~$") and not f.name.startswith(".")]
    print(f"[DataLoader] **/*.docx : found {len(docx_files)} file(s)")
    if docx_files:
        try:
            import docx2txt
            has_docx2txt = True
        except ImportError:
            has_docx2txt = False

        for fpath in docx_files:
            try:
                meta = extract_metadata_from_path(fpath, data_path)
                if has_docx2txt:
                    text = docx2txt.process(str(fpath))
                else:
                    from langchain_community.document_loaders import Docx2txtLoader
                    text = Docx2txtLoader(str(fpath)).load()[0].page_content
                if text and text.strip():
                    documents.append(Document(page_content=text, metadata=meta))
            except Exception as exc:
                print(f"[DataLoader] [Warning] Skipping '{fpath.name}': {exc}")

    print(f"[DataLoader] Total pages/documents loaded: {len(documents)}")
    return documents


if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"\n✅ Loaded {len(docs)} document(s).")
    if docs:
        print("First document metadata:", docs[0].metadata)
        print("Preview:", docs[0].page_content[:200], "...")
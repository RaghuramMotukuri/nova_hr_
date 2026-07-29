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
    }


def load_all_documents(data_dir: str = "data") -> List[Any]:
    """
    Recursively load all supported files from *data_dir*.
    Supported formats: PDF, TXT, DOCX.

    Returns a list of LangChain Document objects (each enriched with
    company, subfolder, filename and source in metadata).
    """
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        Docx2txtLoader,
    )

    data_path = Path(data_dir).resolve()

    if not data_path.exists():
        print(f"[DataLoader] '{data_path}' not found - creating empty directory.")
        data_path.mkdir(parents=True, exist_ok=True)
        return []

    documents: List[Any] = []

    # ── PDFs ─────────────────────────────────────────────────────────────────
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DataLoader] **/*.pdf : found {len(pdf_files)} file(s)")
    for fpath in pdf_files:
        try:
            docs = PyPDFLoader(str(fpath)).load()
            meta = extract_metadata_from_path(fpath, data_path)
            for doc in docs:
                doc.metadata.update(meta)
            documents.extend(docs)
        except Exception as exc:
            print(f"[DataLoader] [Warning] Skipping '{fpath.name}': {exc}")

    # ── TXT files ─────────────────────────────────────────────────────────────
    txt_files = list(data_path.glob("**/*.txt"))
    print(f"[DataLoader] **/*.txt  : found {len(txt_files)} file(s)")
    for fpath in txt_files:
        try:
            # Explicit encoding prevents UnicodeDecodeError on Windows
            docs = TextLoader(str(fpath), encoding="utf-8").load()
            meta = extract_metadata_from_path(fpath, data_path)
            for doc in docs:
                doc.metadata.update(meta)
            documents.extend(docs)
        except Exception as exc:
            print(f"[DataLoader] [Warning] Skipping '{fpath.name}': {exc}")

    # ── DOCX files ────────────────────────────────────────────────────────────
    docx_files = list(data_path.glob("**/*.docx"))
    print(f"[DataLoader] **/*.docx : found {len(docx_files)} file(s)")
    for fpath in docx_files:
        try:
            docs = Docx2txtLoader(str(fpath)).load()
            meta = extract_metadata_from_path(fpath, data_path)
            for doc in docs:
                doc.metadata.update(meta)
            documents.extend(docs)
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
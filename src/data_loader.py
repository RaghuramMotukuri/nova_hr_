"""
data_loader.py
Load HR policy documents (PDF, TXT, DOCX) from the data/ directory.
Applies full sanitization via preprocessor.sanitize_chunk() on every chunk.
Extracts company / subfolder metadata from path structure.
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

try:
    from .preprocessor import sanitize_chunk, is_junk_chunk
except ImportError:
    from src.preprocessor import sanitize_chunk, is_junk_chunk


def extract_metadata_from_path(fpath: Path | str, data_dir: Path | str) -> Dict[str, Any]:
    """Extract company, subfolder, filename from path structure: data/company/subfolder/file."""
    fpath = Path(fpath)
    data_dir = Path(data_dir)
    try:
        relative = fpath.resolve().relative_to(data_dir.resolve())
        parts = relative.parts
        if len(parts) >= 3:
            company, subfolder, filename = parts[0], "/".join(parts[1:-1]), parts[-1]
        elif len(parts) == 2:
            company, subfolder, filename = parts[0], "General", parts[-1]
        else:
            company, subfolder, filename = "General", "General", fpath.name
    except Exception:
        company, subfolder, filename = "General", "General", fpath.name

    subtopic = subfolder if subfolder != "General" else fpath.stem
    return {
        "company": company, "company_id": company,
        "subfolder": subfolder, "subtopic": subtopic,
        "filename": filename, "document_id": filename,
        "source": str(fpath.resolve()), "source_file": str(fpath.resolve()),
        "page": 1, "page_number": 1, "section_header": "",
    }


def _load_pdfs(data_path: Path) -> List[Any]:
    from langchain_core.documents import Document
    docs: List[Any] = []
    files = [f for f in data_path.glob("**/*.pdf") if not f.name.startswith(("~$", "."))]
    print(f"[DataLoader] PDF: found {len(files)} file(s)")
    if not files:
        return docs
    try:
        import fitz
        for fpath in files:
            meta = extract_metadata_from_path(fpath, data_path)
            try:
                pdf = fitz.open(str(fpath))
                for pg_idx, page in enumerate(pdf, start=1):
                    raw_text = page.get_text("text")
                    text = sanitize_chunk(raw_text)
                    if not text or is_junk_chunk(text):
                        continue
                    m = {**meta, "page": pg_idx, "page_number": pg_idx}
                    docs.append(Document(page_content=text, metadata=m))
                pdf.close()
            except Exception as exc:
                print(f"[DataLoader] PDF parse error ({fpath.name}): {exc}")
    except ImportError:
        print("[DataLoader] PyMuPDF not available, skipping PDFs.")
    return docs


def _load_txts(data_path: Path) -> List[Any]:
    from langchain_core.documents import Document
    docs: List[Any] = []
    files = [f for f in data_path.glob("**/*.txt") if not f.name.startswith(("~$", "."))]
    print(f"[DataLoader] TXT: found {len(files)} file(s)")
    for fpath in files:
        meta = extract_metadata_from_path(fpath, data_path)
        try:
            raw = fpath.read_text(encoding="utf-8", errors="ignore")
            text = sanitize_chunk(raw)
            if text and not is_junk_chunk(text):
                docs.append(Document(page_content=text, metadata=meta))
        except Exception as exc:
            print(f"[DataLoader] TXT parse error ({fpath.name}): {exc}")
    return docs


def _load_docx(data_path: Path) -> List[Any]:
    from langchain_core.documents import Document
    docs: List[Any] = []
    files = [f for f in data_path.glob("**/*.docx") if not f.name.startswith(("~$", "."))]
    print(f"[DataLoader] DOCX: found {len(files)} file(s)")
    if not files:
        return docs
    try:
        import docx2txt
        for fpath in files:
            meta = extract_metadata_from_path(fpath, data_path)
            try:
                raw = docx2txt.process(str(fpath))
                text = sanitize_chunk(raw)
                if text and not is_junk_chunk(text):
                    docs.append(Document(page_content=text, metadata=meta))
            except Exception as exc:
                print(f"[DataLoader] DOCX parse error ({fpath.name}): {exc}")
    except ImportError:
        print("[DataLoader] docx2txt not available, skipping DOCX.")
    return docs


def load_all_documents(data_dir: str = "data") -> List[Any]:
    """
    Recursively load all PDF, TXT, DOCX files from data_dir.
    Every chunk is sanitized (index markers stripped) via preprocessor.
    Returns a list of LangChain Document objects.
    """
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        print(f"[DataLoader] '{data_path}' not found — creating empty directory.")
        data_path.mkdir(parents=True, exist_ok=True)
        return []

    docs = _load_pdfs(data_path) + _load_txts(data_path) + _load_docx(data_path)
    print(f"[DataLoader] Total loaded: {len(docs)} document chunk(s)")
    return docs


# ── In-memory file parsing (for Streamlit uploads) ────────────────────────────

def parse_uploaded_pdf(file_bytes: bytes, filename: str, company: str = "General", subfolder: str = "General") -> List[Any]:
    """Parse a PDF from raw bytes (in-memory, no disk write)."""
    from langchain_core.documents import Document
    docs: List[Any] = []
    meta_base = {
        "company": company, "company_id": company,
        "subfolder": subfolder, "subtopic": subfolder,
        "filename": filename, "document_id": filename,
        "source": f"upload://{company}/{subfolder}/{filename}", "source_file": f"upload://{company}/{subfolder}/{filename}",
        "section_header": "",
    }
    try:
        import fitz
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        for pg_idx, page in enumerate(pdf, start=1):
            raw_text = page.get_text("text")
            text = sanitize_chunk(raw_text)
            if not text or is_junk_chunk(text):
                continue
            m = {**meta_base, "page": pg_idx, "page_number": pg_idx}
            docs.append(Document(page_content=text, metadata=m))
        pdf.close()
        print(f"[DataLoader] Parsed PDF '{filename}' in-memory: {len(docs)} chunk(s)")
    except ImportError:
        print("[DataLoader] PyMuPDF not available, cannot parse PDF.")
    except Exception as exc:
        print(f"[DataLoader] PDF parse error ({filename}): {exc}")
    return docs


def parse_uploaded_txt(file_bytes: bytes, filename: str, company: str = "General", subfolder: str = "General") -> List[Any]:
    """Parse a TXT file from raw bytes (in-memory, no disk write)."""
    from langchain_core.documents import Document
    docs: List[Any] = []
    meta = {
        "company": company, "company_id": company,
        "subfolder": subfolder, "subtopic": subfolder,
        "filename": filename, "document_id": filename,
        "source": f"upload://{company}/{subfolder}/{filename}", "source_file": f"upload://{company}/{subfolder}/{filename}",
        "page": 1, "page_number": 1, "section_header": "",
    }
    try:
        raw = file_bytes.decode("utf-8", errors="ignore")
        text = sanitize_chunk(raw)
        if text and not is_junk_chunk(text):
            docs.append(Document(page_content=text, metadata=meta))
        print(f"[DataLoader] Parsed TXT '{filename}' in-memory: {len(docs)} chunk(s)")
    except Exception as exc:
        print(f"[DataLoader] TXT parse error ({filename}): {exc}")
    return docs


def parse_uploaded_docx(file_bytes: bytes, filename: str, company: str = "General", subfolder: str = "General") -> List[Any]:
    """Parse a DOCX file from raw bytes (in-memory, no disk write)."""
    from langchain_core.documents import Document
    docs: List[Any] = []
    meta = {
        "company": company, "company_id": company,
        "subfolder": subfolder, "subtopic": subfolder,
        "filename": filename, "document_id": filename,
        "source": f"upload://{company}/{subfolder}/{filename}", "source_file": f"upload://{company}/{subfolder}/{filename}",
        "page": 1, "page_number": 1, "section_header": "",
    }
    try:
        import docx2txt
        raw = docx2txt.process(BytesIO(file_bytes))
        text = sanitize_chunk(raw)
        if text and not is_junk_chunk(text):
            docs.append(Document(page_content=text, metadata=meta))
        print(f"[DataLoader] Parsed DOCX '{filename}' in-memory: {len(docs)} chunk(s)")
    except ImportError:
        print("[DataLoader] docx2txt not available, cannot parse DOCX.")
    except Exception as exc:
        print(f"[DataLoader] DOCX parse error ({filename}): {exc}")
    return docs


def parse_uploaded_files(uploaded_files: list, company: str = "General", subfolder: str = "General") -> List[Any]:
    """
    Parse multiple uploaded Streamlit files in-memory (no disk writes).
    Returns a list of LangChain Document objects ready for embedding.
    """
    all_docs: List[Any] = []
    for f in uploaded_files:
        file_bytes = f.getvalue()
        fname = f.name.lower()
        if fname.endswith(".pdf"):
            all_docs.extend(parse_uploaded_pdf(file_bytes, f.name, company, subfolder))
        elif fname.endswith(".txt"):
            all_docs.extend(parse_uploaded_txt(file_bytes, f.name, company, subfolder))
        elif fname.endswith(".docx"):
            all_docs.extend(parse_uploaded_docx(file_bytes, f.name, company, subfolder))
        else:
            print(f"[DataLoader] Skipping unsupported file type: {f.name}")
    print(f"[DataLoader] Total parsed from upload: {len(all_docs)} chunk(s) [{company}/{subfolder}]")
    return all_docs

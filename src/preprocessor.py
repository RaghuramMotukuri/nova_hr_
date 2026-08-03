"""
preprocessor.py
Dedicated text sanitizer for LOVA HR.

Strips index numbers, page markers, and document noise from input text
BEFORE it enters the embedding or LLM generation pipeline.

Rules (applied in order):
  1. Strip leading index markers: 1., [1], 1), (1), i., (a)
  2. Strip inline citations: [1], (2) inside sentences
  3. Strip page/doc header-footer markers
  4. Normalize whitespace
"""
from __future__ import annotations
import re
from typing import List

# ── Compiled patterns ─────────────────────────────────────────────────────────

_LEADING_INDEX = re.compile(
    r"^\s*(?:"
    r"\[\d+\]|\(\d+\)|\d+\.|\d+\)"
    r"|[ivxlcdm]+\.\s|\([a-z]\)|[a-z]\."
    r")\s*",
    re.IGNORECASE,
)

_INLINE_INDEX = re.compile(r"\s*\[\d+\]|\s*\(\d+\)")

_PAGE_MARKERS = re.compile(
    r"(?i)(?:page\s+\d+\s+of\s+\d+|page\s+\d+|\d+\s+of\s+\d+"
    r"|ver\.?\s*rev\.?|doc\s*id\s*:|document\s+no\.?\s*:"
    r"|revision\s+\d+|confidential\s*[-]?|internal\s+use\s+only"
    r"|draft\s+v\d+)"
)

_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# ── Public API ────────────────────────────────────────────────────────────────

def strip_index_markers(text: str) -> str:
    """Remove leading index markers from each line."""
    return "\n".join(_LEADING_INDEX.sub("", line) for line in text.split("\n"))

def strip_inline_indices(text: str) -> str:
    """Remove inline citation markers [1], (2) from within sentences."""
    return _INLINE_INDEX.sub("", text)

def strip_page_markers(text: str) -> str:
    """Drop lines that are pure page/document markers; remove inline occurrences."""
    cleaned = []
    for line in text.split("\n"):
        if _PAGE_MARKERS.fullmatch(line.strip()):
            continue
        cleaned.append(_PAGE_MARKERS.sub("", line).rstrip())
    return "\n".join(cleaned)

def normalize_whitespace(text: str) -> str:
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()

def sanitize_chunk(text: str) -> str:
    """Full sanitization pipeline for a document chunk."""
    if not text or not text.strip():
        return ""
    text = strip_page_markers(text)
    text = strip_index_markers(text)
    text = strip_inline_indices(text)
    return normalize_whitespace(text)

def sanitize_query(query: str) -> str:
    """Light sanitization for a user query."""
    if not query or not query.strip():
        return ""
    query = strip_inline_indices(query)
    return normalize_whitespace(query)

def is_junk_chunk(text: str) -> bool:
    """True if the chunk is structural noise (TOC, index page, cover page)."""
    if not text or not text.strip():
        return True
    text_lower = text.lower()
    toc_headers = ["table of contents", "index page", "table of content", "contents"]
    if any(h in text_lower for h in toc_headers):
        return True
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True
    dot_leaders = sum(1 for l in lines if re.search(r"\.{3,}\s*\d+", l))
    if len(lines) >= 3 and dot_leaders / len(lines) >= 0.4:
        return True
    numeric_or_page = sum(1 for l in lines if re.search(r"(?:page\s*\d+|\b\d+\b)$", l, re.IGNORECASE))
    if len(lines) >= 3 and numeric_or_page / len(lines) >= 0.5:
        return True
    return len(text.strip()) < 30

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


def clean_conversational_response(text: str) -> str:
    """
    Strips out markdown headings, bullets, asterisks, hashes, backticks,
    brackets, and extra typographical symbols to yield pure, clean human-like prose.
    """
    if not text or not text.strip():
        return ""

    # Remove markdown headings (e.g. ### Summary -> Summary)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove bullet markers at line starts (*, -, +, •, ⁃)
    text = re.sub(r"^\s*[*+\-•⁃]\s*", "", text, flags=re.MULTILINE)

    # Remove numbered list prefixes at line starts (1., 2), a., etc.)
    text = re.sub(r"^\s*\d+[\.\)]\s*", "", text, flags=re.MULTILINE)

    # Remove bold/italic symbols (**word**, *word*, __word__, _word_)
    text = re.sub(r"\*{1,3}|_{1,3}", "", text)

    # Remove backticks and code formatting
    text = re.sub(r"`+", "", text)

    # Remove citation brackets like [1], [Document: ...], [Section: ...], [header]
    text = re.sub(r"\[\d+\]|\[Document:[^\]]*\]|\[Section:[^\]]*\]|\[header\]", "", text)

    # Remove decorative divider lines (---, ===, ***)
    text = re.sub(r"^\s*[-=*]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove robotic document preambles
    text = re.sub(
        r"(?i)^\s*(?:the\s+document\s+states\s+that|according\s+to\s+the\s+document|according\s+to\s+the\s+policy|the\s+policy\s+states\s+that|associates\s+are\s+eligible\s+for\s+a\s+balanced\s+pool\s+of\s+leaves\s+to\s+maintain\s+professional\s+and\s+personal\s+equilibrium:?|based\s+to\s+the\s+(?:official\s+|provided\s+)?policy\s+(?:context|document|documents),?)\s*",
        "",
        text,
    )

    # Strip section labels & category headers (Casual / Friendly Version, Direct Version, Summary, etc.)
    text = re.sub(
        r"(?i)\b(?:casual\s*/?\s*friendly\s+version|casual\s+version|direct\s+version|summary\s+option|summary\s+pitch|summary\s*version|summary|option\b[^:]*:?)\s*:?",
        "",
        text,
    )

    # Remove section numbers like "3 3 Workplace Diversity" or "Section 4.2"
    text = re.sub(r"(?i)\bSection\s+\d+[\.\d]*\s*", "", text)
    text = re.sub(r"\b\d+[\s\.]?\d*\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b", "", text)

    # Remove document title artifacts mid-sentence
    text = re.sub(r"(?i)\bHuman Rights Policy Statement\s*", "", text)
    text = re.sub(r"(?i)\bWorkplace Diversity\s*", "", text)

    # Remove orphaned company prefixes like "For TCS ," or "For Infosys ,"
    text = re.sub(r"(?i)\bFor\s+(?:TCS|Infosys|Wipro|Tata|Accenture|HCL)\s*,?\s*", "", text)

    # Remove duplicate company names that appear consecutively
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)

    # Fix spacing before commas (e.g., "word ," -> "word,")
    text = re.sub(r"\s+,", ",", text)

    # Fix spacing after commas (e.g., "word,word" -> "word, word")
    text = re.sub(r",(\S)", r", \1", text)

    # Translate corporate jargon into friendly human conversational language
    text = re.sub(r"(?i)associates are eligible for a balanced pool of leaves to maintain professional and personal equilibrium:?", "", text)
    text = re.sub(r"(?i)immediate personal interventions", "urgent personal matters", text)
    text = re.sub(r"(?i)repatriation tracking", "a plan for your return", text)
    text = re.sub(r"(?i)accrued month-on-month", "earned each month", text)
    text = re.sub(r"(?i)requiring structured manager sign-off", "with manager approval", text)
    text = re.sub(r"(?i)allocated for unexpected medical occurrences", "set aside for medical needs", text)
    text = re.sub(r"(?i)provided to high-performing long-term associates", "offered to eligible team members", text)

    # Replace multiple newlines with standard sentence spacing
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned = " ".join(lines)

    # Normalize whitespace
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    # Remove leading/trailing punctuation or orphaned words
    cleaned = re.sub(r"^[,\s]+", "", cleaned)
    cleaned = re.sub(r"[,\s]+$", "", cleaned)

    return cleaned


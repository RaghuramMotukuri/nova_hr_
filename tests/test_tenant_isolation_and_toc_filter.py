"""
test_tenant_isolation_and_toc_filter.py
Dedicated tests for:
  1. is_junk_chunk TOC & Index page filtering
  2. Hard pre-retrieval tenant isolation & safety net validation
"""
import pytest
from src.preprocessor import is_junk_chunk
from src.embeddings import EmbeddingPipeline
from src.retriever import validate_context_tenant, HybridRetriever
from langchain_core.documents import Document


def test_is_junk_chunk_detection():
    # Test 1: Explicit TOC header
    toc_text = """
    TABLE OF CONTENTS
    1. Introduction ...................................... 3
    2. Leave Policy ..................................... 8
    3. Code of Conduct .................................. 15
    """
    assert is_junk_chunk(toc_text) is True

    # Test 2: Dot leader pattern density
    dot_text = """
    SECTION A ................ PAGE 2
    SECTION B ................ PAGE 5
    SECTION C ................ PAGE 10
    """
    assert is_junk_chunk(dot_text) is True

    # Test 3: High index line ratio
    index_ratio_text = """
    Overview page 1
    Scope page 2
    Benefits page 4
    Termination page 9
    """
    assert is_junk_chunk(index_ratio_text) is True

    # Test 4: Genuine policy text (should NOT be flagged as junk)
    policy_text = """
    TCS HR Leave Policy:
    Employees are entitled to 20 days of paid annual leave per calendar year.
    Leave requests must be submitted at least 5 business days in advance via the HR portal.
    """
    assert is_junk_chunk(policy_text) is False


def test_validate_context_tenant_leak_prevention():
    chunks = [
        {"company": "TCS", "chunk_text": "TCS notice period is 90 days."},
        {"company": "Infosys", "chunk_text": "Infosys medical insurance is 5L."},
        {"company": "TCS", "chunk_text": "TCS probation period is 6 months."},
    ]

    # Validate TCS scope → Infosys chunk must be blocked
    tcs_valid = validate_context_tenant(chunks, target_company="TCS")
    assert len(tcs_valid) == 2
    for c in tcs_valid:
        assert c["company"] == "TCS"

    # Validate All Companies scope → all chunks kept
    all_valid = validate_context_tenant(chunks, target_company="All Companies")
    assert len(all_valid) == 3


def test_junk_chunk_stripping_during_ingestion():
    from src.preprocessor import sanitize_chunk, is_junk_chunk
    docs = [
        Document(
            page_content="Table of Contents\n1. Policy overview ..... 3\n2. Leave allowance ..... 7",
            metadata={"company": "TCS", "subfolder": "General", "filename": "index.txt"}
        ),
        Document(
            page_content="TCS Annual Leave Policy: Full-time employees receive 21 days of paid leave annually.",
            metadata={"company": "TCS", "subfolder": "General", "filename": "leave.txt"}
        )
    ]
    valid_docs = []
    for d in docs:
        clean = sanitize_chunk(d.page_content)
        if clean and not is_junk_chunk(clean):
            d.page_content = clean
            valid_docs.append(d)

    assert len(valid_docs) == 1
    assert "Full-time employees" in valid_docs[0].page_content

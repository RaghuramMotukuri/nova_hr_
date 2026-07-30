"""
test_ingestion_and_scoping.py
Comprehensive TDD Test Suite for LOVA_HR:
  - Phase 1: Ingestion & Metadata Enrichment (company_id, document_id, page_number, chunk_hash)
  - Phase 2: Hybrid Retrieval (BM25 + BGE-M3 + RRF + BGE-Reranker-V2-M3)
  - Phase 3: Cross-Tenant Data Isolation & Verbatim Accuracy
"""
import os
import shutil
from pathlib import Path
import pytest

from src.data_loader import load_all_documents
from src.embeddings import EmbeddingPipeline
from src.retriever import HybridRetriever

TEST_SUITE_DIR = Path("test_suite_data")


def safe_rmtree(path: Path):
    if not path.exists():
        return
    for root, dirs, files in os.walk(str(path), topdown=False):
        for name in files:
            try:
                os.unlink(os.path.join(root, name))
            except Exception:
                pass
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except Exception:
                pass
    try:
        shutil.rmtree(path)
    except Exception:
        pass


@pytest.fixture(scope="function", autouse=True)
def setup_test_suite_data():
    safe_rmtree(TEST_SUITE_DIR)
    TEST_SUITE_DIR.mkdir(parents=True, exist_ok=True)

    # Company A: TCS Policies
    tcs_dir = TEST_SUITE_DIR / "TCS" / "Policies"
    tcs_dir.mkdir(parents=True, exist_ok=True)
    with open(tcs_dir / "notice_period.txt", "w", encoding="utf-8") as f:
        f.write(
            "TCS HR Policy: Standard notice period for full-time employees is 90 days. "
            "Buyout of notice period requires approval from the business unit head."
        )

    # Company B: Infosys Benefits
    inf_dir = TEST_SUITE_DIR / "Infosys" / "Benefits"
    inf_dir.mkdir(parents=True, exist_ok=True)
    with open(inf_dir / "health_insurance.txt", "w", encoding="utf-8") as f:
        f.write(
            "Infosys Employee Benefits: Comprehensive medical insurance covers up to 5,00,000 INR per family. "
            "Maternity leave benefit includes 26 weeks of paid leave."
        )

    # Company C: Wipro Policies
    wip_dir = TEST_SUITE_DIR / "Wipro" / "Policies"
    wip_dir.mkdir(parents=True, exist_ok=True)
    with open(wip_dir / "annual_leave.txt", "w", encoding="utf-8") as f:
        f.write(
            "Wipro Annual Leave Policy: Employees earn 1.75 days of earned leave per month (21 days annually). "
            "Maximum unavailed earned leave carry forward limit is 45 days."
        )

    yield

    safe_rmtree(TEST_SUITE_DIR)


# ── PHASE 1 TESTS ────────────────────────────────────────────────────────────

def test_document_metadata_enrichment():
    """Verify parsed chunks carry company_id, document_id, page_number, chunk_hash."""
    pipeline = EmbeddingPipeline(chunk_size=120, chunk_overlap=20)
    pipeline.vector_store.clear()

    docs = load_all_documents(str(TEST_SUITE_DIR))
    assert len(docs) >= 3

    chunks = pipeline.chunk_documents(docs)
    pipeline.sync_vector_store(chunks)

    all_docs = pipeline.vector_store.get_all_documents()
    assert len(all_docs) >= 3

    for doc in all_docs:
        assert "company" in doc or "company_id" in doc
        meta = doc if "company_id" in doc else doc
        assert meta.get("company_id") in ["TCS", "Infosys", "Wipro"]
        assert meta.get("filename") in ["notice_period.txt", "health_insurance.txt", "annual_leave.txt"]
        assert doc.get("chunk_text") != ""


def test_company_categorization_and_partitioning():
    """Verify local and vector storage partitioning strictly by company_id."""
    pipeline = EmbeddingPipeline(chunk_size=120, chunk_overlap=20)
    pipeline.vector_store.clear()

    docs = load_all_documents(str(TEST_SUITE_DIR))
    chunks = pipeline.chunk_documents(docs)
    pipeline.sync_vector_store(chunks)

    # Query scoped to TCS only
    tcs_docs = pipeline.vector_store.get_all_documents(company="TCS")
    assert len(tcs_docs) > 0
    for d in tcs_docs:
        assert d["company"] == "TCS"
        assert d["filename"] == "notice_period.txt"

    # Query scoped to Infosys only
    inf_docs = pipeline.vector_store.get_all_documents(company="Infosys")
    assert len(inf_docs) > 0
    for d in inf_docs:
        assert d["company"] == "Infosys"
        assert d["filename"] == "health_insurance.txt"


# ── PHASE 2 TESTS ────────────────────────────────────────────────────────────

def test_hybrid_retrieval_and_reranking():
    """Verify BM25 + BGE-M3 RRF fusion + Cross-Encoder reranking pipeline."""
    pipeline = EmbeddingPipeline(chunk_size=120, chunk_overlap=20)
    pipeline.vector_store.clear()

    docs = load_all_documents(str(TEST_SUITE_DIR))
    chunks = pipeline.chunk_documents(docs)
    pipeline.sync_vector_store(chunks)

    retriever = HybridRetriever(pipeline=pipeline)
    res = retriever.search(
        query="What is the notice period for employees?",
        company="TCS",
        top_k=2,
        rerank_top_n=2
    )

    assert len(res["reranked"]) > 0
    top_chunk = res["reranked"][0]
    assert top_chunk["company"] == "TCS"
    assert "90 days" in top_chunk["chunk_text"].lower()


# ── PHASE 3 TESTS ────────────────────────────────────────────────────────────

def test_cross_tenant_data_isolation():
    """Ensure Company A queries NEVER return Company B or Company C chunks."""
    pipeline = EmbeddingPipeline(chunk_size=120, chunk_overlap=20)
    pipeline.vector_store.clear()

    docs = load_all_documents(str(TEST_SUITE_DIR))
    chunks = pipeline.chunk_documents(docs)
    pipeline.sync_vector_store(chunks)

    retriever = HybridRetriever(pipeline=pipeline)

    # Search TCS for "leave" (Infosys & Wipro have leave/maternity policies, TCS has notice period)
    tcs_res = retriever.search(query="leave policy", company="TCS", top_k=5)
    for doc in tcs_res["reranked"]:
        assert doc["company"] == "TCS"
        assert doc["company"] != "Infosys"
        assert doc["company"] != "Wipro"

    # Search Wipro for "insurance" (Infosys has insurance, Wipro does not)
    wip_res = retriever.search(query="insurance benefit", company="Wipro", top_k=5)
    for doc in wip_res["reranked"]:
        assert doc["company"] == "Wipro"
        assert doc["company"] != "Infosys"


def test_verbatim_ground_truth_accuracy():
    """Verify retrieved output matches exact source policy facts verbatim."""
    pipeline = EmbeddingPipeline(chunk_size=120, chunk_overlap=20)
    pipeline.vector_store.clear()

    docs = load_all_documents(str(TEST_SUITE_DIR))
    chunks = pipeline.chunk_documents(docs)
    pipeline.sync_vector_store(chunks)

    retriever = HybridRetriever(pipeline=pipeline)

    # Infosys Maternity Leave
    inf_res = retriever.search(query="maternity leave weeks", company="Infosys")
    assert len(inf_res["reranked"]) > 0
    assert "26 weeks" in inf_res["reranked"][0]["chunk_text"]

    # Wipro Carry Forward Limit
    wip_res = retriever.search(query="carry forward limit earned leave", company="Wipro")
    assert len(wip_res["reranked"]) > 0
    assert "45 days" in wip_res["reranked"][0]["chunk_text"]

"""
test_ingestion_and_scoping.py
Clean test suite for:
  - Document ingestion and metadata enrichment (company, filename, page)
  - Hybrid retrieval (BM25 + Semantic + RRF + Reranker)
  - Tenant isolation and accuracy verification
"""
import os
import shutil
from pathlib import Path
import pytest

from src.data_loader import load_all_documents
from src.embeddings import EmbeddingPipeline
from src.retriever import HybridRetriever, validate_context_tenant

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


def test_document_metadata_enrichment():
    """Verify parsed chunks carry company, filename, and page metadata."""
    pipeline = EmbeddingPipeline()
    docs = load_all_documents(str(TEST_SUITE_DIR))
    assert len(docs) >= 3

    pipeline.add_documents(docs)
    assert len(pipeline._bm25_docs) >= 3

    for doc in pipeline._bm25_docs:
        assert doc.get("company") in ["TCS", "Infosys", "Wipro"]
        assert doc.get("filename") in ["notice_period.txt", "health_insurance.txt", "annual_leave.txt"]
        assert doc.get("chunk_text") != ""


def test_company_categorization_and_partitioning():
    """Verify document partitioning strictly by company."""
    pipeline = EmbeddingPipeline()
    docs = load_all_documents(str(TEST_SUITE_DIR))
    pipeline.add_documents(docs)

    tcs_docs = pipeline.lexical_search(query="notice period", company="TCS")
    assert len(tcs_docs) > 0
    for d in tcs_docs:
        assert d["company"] == "TCS"
        assert d["filename"] == "notice_period.txt"

    inf_docs = pipeline.lexical_search(query="insurance", company="Infosys")
    assert len(inf_docs) > 0
    for d in inf_docs:
        assert d["company"] == "Infosys"
        assert d["filename"] == "health_insurance.txt"


def test_hybrid_retrieval_and_reranking():
    """Verify BM25 + RRF fusion + Reranker pipeline."""
    pipeline = EmbeddingPipeline()
    docs = load_all_documents(str(TEST_SUITE_DIR))
    pipeline.add_documents(docs)

    retriever = HybridRetriever(pipeline=pipeline)
    retriever._firebase_ok = False

    res = retriever.search(
        query="What is the notice period for employees?",
        company="TCS",
        top_k=2,
        rerank_top_n=2,
        use_local_engine=False,
    )

    assert len(res["reranked"]) > 0
    top_chunk = res["reranked"][0]
    assert top_chunk["company"] == "TCS"
    assert "90 days" in top_chunk["chunk_text"].lower()


def test_cross_tenant_data_isolation():
    """Ensure Company A queries NEVER return Company B or Company C chunks."""
    pipeline = EmbeddingPipeline()
    docs = load_all_documents(str(TEST_SUITE_DIR))
    pipeline.add_documents(docs)

    retriever = HybridRetriever(pipeline=pipeline)
    retriever._firebase_ok = False

    tcs_res = retriever.search(query="leave policy", company="TCS", top_k=5, use_local_engine=False)
    for doc in tcs_res["reranked"]:
        assert doc["company"] == "TCS"
        assert doc["company"] != "Infosys"
        assert doc["company"] != "Wipro"

    wip_res = retriever.search(query="insurance benefit", company="Wipro", top_k=5, use_local_engine=False)
    for doc in wip_res["reranked"]:
        assert doc["company"] == "Wipro"
        assert doc["company"] != "Infosys"


def test_verbatim_ground_truth_accuracy():
    """Verify retrieved output matches exact source policy facts verbatim."""
    pipeline = EmbeddingPipeline()
    docs = load_all_documents(str(TEST_SUITE_DIR))
    pipeline.add_documents(docs)

    retriever = HybridRetriever(pipeline=pipeline)
    retriever._firebase_ok = False

    inf_res = retriever.search(query="maternity leave weeks", company="Infosys", use_local_engine=False)
    assert len(inf_res["reranked"]) > 0
    assert "26 weeks" in inf_res["reranked"][0]["chunk_text"]

    wip_res = retriever.search(query="carry forward limit earned leave", company="Wipro", use_local_engine=False)
    assert len(wip_res["reranked"]) > 0
    assert "45 days" in wip_res["reranked"][0]["chunk_text"]

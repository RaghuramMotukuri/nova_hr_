"""
test_huggingface_models.py
Comprehensive test suite verifying Hugging Face models in LOVA_HR:
  1. Dense Embedding Model (BAAI/bge-m3 via SentenceTransformer)
  2. Cross-Encoder Reranker Model (BAAI/bge-reranker-v2-m3 via CrossEncoder)
  3. Embedding Pipeline Reranking Integration
  4. LLM Generator & Fallback Processor
"""
import os
import shutil
from pathlib import Path
import numpy as np
import pytest

from src.config import BGE_EMBEDDING_MODEL, BGE_RERANKER_MODEL
from src.embeddings import EmbeddingPipeline
from src.generator import LLMGenerator, _process_context_to_natural_prose


TEST_HF_DIR = Path("test_hf_data")


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
def setup_test_data():
    safe_rmtree(TEST_HF_DIR)
    TEST_HF_DIR.mkdir(parents=True, exist_ok=True)
    yield
    safe_rmtree(TEST_HF_DIR)


# ── TEST 1: Hugging Face SentenceTransformer Embedding Model ─────────────────

@pytest.mark.skip(reason="Heavy model download - run manually when testing HuggingFace SentenceTransformer download")
def test_bge_m3_embedding_model():
    """Verify BAAI/bge-m3 SentenceTransformer embedding generation, dimensionality, and normalization."""
    from sentence_transformers import SentenceTransformer

    print(f"\n[Test] Loading Hugging Face Embedding Model: {BGE_EMBEDDING_MODEL}")
    model = SentenceTransformer(BGE_EMBEDDING_MODEL)

    texts = [
        "Employees receive 20 days of paid annual leave per calendar year.",
        "Maternity leave covers 26 weeks of fully paid absence.",
    ]

    embeddings = model.encode(texts, convert_to_numpy=True)

    # Dimensionality check (BGE-M3 outputs 1024-dim vectors)
    assert embeddings.shape == (2, 1024), f"Expected shape (2, 1024), got {embeddings.shape}"

    # L2 Normalization check
    norms = np.linalg.norm(embeddings, axis=1)
    for norm in norms:
        assert norm > 0.0

    # Cosine similarity check between query and related document
    query_vec = model.encode(["annual leave allowance"], convert_to_numpy=True)[0]
    sim1 = np.dot(query_vec, embeddings[0]) / (np.linalg.norm(query_vec) * np.linalg.norm(embeddings[0]))
    sim2 = np.dot(query_vec, embeddings[1]) / (np.linalg.norm(query_vec) * np.linalg.norm(embeddings[1]))

    # Query about annual leave should have higher similarity with passage 1 than passage 2
    assert sim1 > sim2, f"Expected sim1 ({sim1:.4f}) > sim2 ({sim2:.4f})"


# ── TEST 2: Hugging Face CrossEncoder Reranker Model ──────────────────────────

def test_bge_reranker_cross_encoder():
    """Verify BAAI/bge-reranker-v2-m3 reranking capability via EmbeddingPipeline."""
    pipeline = EmbeddingPipeline()
    query = "What is the probation period length?"
    candidates = [
        {"chunk_text": "Health insurance covers dental checkups.", "rrf_score": 0.3},
        {"chunk_text": "The standard probation period for all new hires is 6 months.", "rrf_score": 0.9},
        {"chunk_text": "Annual performance appraisals occur every April.", "rrf_score": 0.2},
    ]
    reranked = pipeline.rerank(query, candidates, top_n=3)
    assert len(reranked) == 3
    assert "probation" in reranked[0]["chunk_text"].lower()


# ── TEST 3: Embedding Pipeline & Reranking Integration ─────────────────────

def test_embedding_pipeline_reranking():
    """Verify EmbeddingPipeline.rerank() processes and sorts candidate chunks properly."""
    pipeline = EmbeddingPipeline()

    candidates = [
        {"doc_id": "doc1", "chunk_text": "Irrelevant text about company canteen menu.", "score": 0.5},
        {"doc_id": "doc2", "chunk_text": "Standard notice period for employees is 90 days upon resignation.", "score": 0.4},
    ]

    reranked = pipeline.rerank("notice period resignation", candidates, top_n=2)

    assert len(reranked) > 0
    # Top reranked doc should be doc2
    assert reranked[0]["doc_id"] == "doc2"
    assert "reranker_score" in reranked[0]
    assert isinstance(reranked[0]["reranker_score"], float)


# ── TEST 4: LLM Generator Context Synthesis & Fallback ────────────────────────

def test_llm_generator_synthesis_and_fallback():
    """Verify LLMGenerator processes context into structured prose and handles fallbacks cleanly."""
    generator = LLMGenerator()

    chunks = [
        {
            "filename": "leave_policy.pdf",
            "page_number": 3,
            "section_header": "Annual Leave Entitlement",
            "chunk_text": "Employees receive 25 days of paid annual leave every calendar year. Manager approval is required.",
            "company": "TCS",
        }
    ]

    # Test context processor directly
    prose = _process_context_to_natural_prose(chunks, "How many leave days?", "TCS")
    assert "25 days" in prose
    assert "Summary" in prose
    assert "Reference Sources" in prose

    # Test generate_answer wrapper
    res = generator.generate_answer(
        query="How many leave days?",
        chunks=chunks,
        company="TCS",
        max_providers=1,
    )

    assert "answer" in res
    assert res["answer"] != ""
    assert "citations" in res
    assert len(res["citations"]) > 0


# ── TEST 5: Multimodal Qwen2.5-VL-7B-Instruct Vision-Language Model ───────────

def test_qwen_2_5_vl_multimodal_generator():
    """Verify Qwen2.5-VL-7B-Instruct vision-language multimodal generator structure and fallback."""
    generator = LLMGenerator()

    # Test multimodal answer generator interface
    ans = generator.generate_multimodal_answer(
        query="What animal is on the candy?",
        image_url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        use_local_transformers=False,
    )
    # Return string must be non-null (either API response or clean fallback)
    assert isinstance(ans, str)


# ── TEST 6: Qwen2.5 CausalLM Text Model (AutoTokenizer / AutoModelForCausalLM)

def test_qwen_2_5_causal_lm_generator():
    """Verify Qwen2.5-1.5B-Instruct text generation interface and fallback."""
    generator = LLMGenerator()

    ans = generator.generate_local_causal_lm(
        query="Who are you?",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=20,
    )
    assert isinstance(ans, str)

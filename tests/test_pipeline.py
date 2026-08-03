"""
test_pipeline.py
Comprehensive test suite for the rebuilt LOVA HR pipeline.

Tests:
  1. Preprocessor — index stripping, query sanitization, junk detection
  2. BGE-M3 embeddings
  3. BGE Reranker cross-encoder
  4. LLMGenerator — synthesis and topic boundary enforcement
  5. LLMGenerator — local Qwen2.5-1.5B-Instruct
  6. End-to-end HybridRetriever pipeline
"""
import sys
import os
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# 1. Preprocessor Tests
# =============================================================================

class TestPreprocessor:
    def test_strip_leading_numbered_index(self):
        from src.preprocessor import strip_index_markers
        assert strip_index_markers("1. Employees get 20 days leave.") == "Employees get 20 days leave."

    def test_strip_bracket_index(self):
        from src.preprocessor import strip_index_markers
        assert strip_index_markers("[2] See section 4.") == "See section 4."

    def test_strip_paren_index(self):
        from src.preprocessor import strip_index_markers
        assert strip_index_markers("(3) Annual Leave Policy") == "Annual Leave Policy"

    def test_strip_roman_numeral_index(self):
        from src.preprocessor import strip_index_markers
        result = strip_index_markers("i. Introduction")
        assert result.strip() == "Introduction"

    def test_strip_inline_citations(self):
        from src.preprocessor import strip_inline_indices
        result = strip_inline_indices("Leave is 20 days [1].")
        assert "[1]" not in result
        assert "Leave is 20 days" in result

    def test_strip_page_marker_full_line(self):
        from src.preprocessor import strip_page_markers
        result = strip_page_markers("Page 1 of 10\nThis is policy text.")
        assert "Page 1 of 10" not in result
        assert "This is policy text." in result

    def test_sanitize_chunk_full_pipeline(self):
        from src.preprocessor import sanitize_chunk
        raw = "[1] Employees get 20 days leave.\n2. Annual leave policy.\nPage 3 of 10"
        result = sanitize_chunk(raw)
        assert "[1]" not in result
        assert "Page 3 of 10" not in result
        assert "Employees get 20 days leave" in result

    def test_sanitize_query_preserves_words(self):
        from src.preprocessor import sanitize_query
        q = "Does this policy [1] require additional approvals?"
        result = sanitize_query(q)
        assert "[1]" not in result
        assert "require additional approvals" in result

    def test_is_junk_chunk_toc(self):
        from src.preprocessor import is_junk_chunk
        assert is_junk_chunk("Table of Contents\n1. Leave Policy......10\n2. Benefits.......20")

    def test_is_junk_chunk_real_content(self):
        from src.preprocessor import is_junk_chunk
        assert not is_junk_chunk("Employees are entitled to twenty days of paid annual leave per calendar year.")


# =============================================================================
# 2. BGE-M3 Embedding Tests
# =============================================================================

class TestBGEEmbeddings:
    def test_bge_m3_produces_embeddings(self):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3")
        vecs = model.encode(["Annual leave policy for employees."])
        assert len(vecs) == 1
        assert len(vecs[0]) == 1024

    def test_bge_m3_sanitized_text_embedding(self):
        from sentence_transformers import SentenceTransformer
        from src.preprocessor import sanitize_chunk
        model = SentenceTransformer("BAAI/bge-m3")
        raw = "1. Employees get 20 days leave [1]. Page 2 of 10"
        clean = sanitize_chunk(raw)
        vecs = model.encode([clean])
        assert len(vecs[0]) == 1024


# =============================================================================
# 3. BGE Reranker Tests
# =============================================================================

class TestBGEReranker:
    def test_embedding_pipeline_rerank(self):
        from src.embeddings import EmbeddingPipeline
        pipeline = EmbeddingPipeline()
        candidates = [
            {"chunk_text": "Employees receive 20 days of paid annual leave.", "rrf_score": 0.8},
            {"chunk_text": "The company was founded in 1995.", "rrf_score": 0.2},
        ]
        reranked = pipeline.rerank("annual leave allowance", candidates, top_n=2)
        assert len(reranked) == 2
        assert "annual leave" in reranked[0]["chunk_text"].lower()


# =============================================================================
# 4. LLM Generator Tests
# =============================================================================

class TestLLMGenerator:
    SAMPLE_CHUNKS = [
        {
            "filename": "leave_policy.pdf", "page_number": 2,
            "section_header": "Annual Leave",
            "chunk_text": "1. Employees get 20 days paid leave per year. [1] See HR manual.",
            "company": "TCS",
        }
    ]

    def test_synthesis_fallback_produces_answer(self):
        from src.generator import LLMGenerator
        g = LLMGenerator()
        result = g.generate_answer("How many leave days?", self.SAMPLE_CHUNKS, company="TCS")
        assert result.get("answer")
        assert len(result["answer"]) > 10

    def test_answer_has_no_index_markers(self):
        from src.generator import LLMGenerator
        g = LLMGenerator()
        result = g.generate_answer("How many leave days?", self.SAMPLE_CHUNKS, company="TCS")
        answer = result.get("answer", "")
        import re
        assert not re.search(r"^\s*\d+\.\s", answer, re.MULTILINE), "Answer should not start with index numbers"

    def test_out_of_scope_query_refused(self):
        from src.generator import LLMGenerator
        g = LLMGenerator()
        result = g.generate_answer(
            "What is the capital of France?",
            self.SAMPLE_CHUNKS, company="TCS"
        )
        answer = result.get("answer", "").lower()
        assert "outside" in answer or "not available" in answer or "scope" in answer, \
            "Out-of-scope questions should be refused"

    def test_empty_chunks_returns_no_docs_message(self):
        from src.generator import LLMGenerator
        g = LLMGenerator()
        result = g.generate_answer("What is the leave policy?", [], company="TCS")
        assert "No relevant" in result.get("answer", "")

    def test_citations_extracted(self):
        from src.generator import LLMGenerator
        g = LLMGenerator()
        result = g.generate_answer("How many leave days?", self.SAMPLE_CHUNKS, company="TCS")
        assert isinstance(result.get("citations"), list)


# =============================================================================
# 5. Local CausalLM Generator Test
# =============================================================================

class TestLocalCausalLM:
    @pytest.mark.skip(reason="Heavy local model download - run explicitly when testing local 1.5B CausalLM")
    def test_local_qwen_causal_lm(self):
        from src.generator import LLMGenerator
        g = LLMGenerator()
        context = "Employees are entitled to 20 days of paid annual leave per year."
        result = g._call_local_causal_lm(
            query="How many days of leave do employees get?",
            context=context,
            scope="TCS",
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            max_new_tokens=80,
        )
        assert isinstance(result, str)
        print(f"[LocalCausalLM] Response: {result[:200]}")


# =============================================================================
# 6. End-to-End HybridRetriever Test
# =============================================================================

class TestHybridRetriever:
    SAMPLE_CHUNKS = [
        {
            "filename": "leave_policy.pdf", "page_number": 2,
            "section_header": "Annual Leave",
            "chunk_text": "Employees are entitled to 20 days of paid annual leave per year.",
            "company": "TCS", "doc_id": "leave_policy.pdf_2_0",
        }
    ]

    def test_retriever_bm25_search(self):
        from src.retriever import HybridRetriever
        from src.embeddings import EmbeddingPipeline
        pipeline = EmbeddingPipeline()
        pipeline.build_bm25_index(self.SAMPLE_CHUNKS)
        retriever = HybridRetriever(pipeline=pipeline)
        retriever._firebase_ok = False  # Force BM25-only
        result = retriever.search("annual leave days", company="TCS", use_local_engine=False)
        assert result["mode"] == "bm25_only"
        assert isinstance(result["generation"], dict)
        assert result["generation"].get("answer")

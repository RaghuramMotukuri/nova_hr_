"""
retriever.py
HybridRetriever — clean orchestration of the full LOVA HR search pipeline.

Steps:
  1. BM25 Lexical Search         (always available, fast)
  2. FAISS Semantic Search       (in-memory, fast if loaded)
  3. RRF Hybrid Fusion           (combine lexical + semantic)
  4. LLM Answer Generation       (local Qwen2.5 or Cloud HF)
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

try:
    from .embeddings import EmbeddingPipeline
    from .config import is_firebase_available
    from .generator import LLMGenerator
    from .preprocessor import sanitize_query
except ImportError:
    from src.embeddings import EmbeddingPipeline
    from src.config import is_firebase_available
    from src.generator import LLMGenerator
    from src.preprocessor import sanitize_query


def validate_context_tenant(
    chunks: List[Dict[str, Any]], target_company: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Safety net: hard-blocks chunks from other companies before returning to UI/LLM.
    """
    if not target_company or target_company == "All Companies":
        return chunks
    valid = []
    for chunk in chunks:
        chunk_company = chunk.get("company", chunk.get("company_id", ""))
        if chunk_company and chunk_company != target_company:
            print(f"[HybridRetriever] LEAK PREVENTED: blocked chunk from '{chunk_company}' (target: '{target_company}')")
            continue
        valid.append(chunk)
    return valid


class HybridRetriever:
    """
    Orchestrates the full retrieval and answer generation pipeline.
    Uses BM25 + FAISS for fast local search.
    """

    def __init__(self, pipeline: Optional[EmbeddingPipeline] = None):
        self.pipeline = pipeline or EmbeddingPipeline()
        self.generator = LLMGenerator()
        self._firebase_ok: Optional[bool] = None
        self.faiss_index = None  # Set by app.py for fast vector search

    def _check_firebase(self) -> bool:
        if self._firebase_ok is None:
            self._firebase_ok = is_firebase_available()
            if not self._firebase_ok:
                print("[HybridRetriever] Firebase unavailable — BM25-only mode.")
        return self._firebase_ok

    def reset_firebase_status(self) -> None:
        self._firebase_ok = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        company: Optional[str] = None,
        subfolder: Optional[str] = None,
        rerank_top_n: int = 5,
        semantic_limit: int = 20,
        lexical_limit: int = 20,
        max_providers: int = 1,
        rag_mode: str = "hybrid",
        hf_model: str = "Qwen/Qwen2.5-3B-Instruct",
        use_local_engine: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full retrieval + answer generation pipeline.
        Uses BM25 + FAISS for fast local search.
        """
        clean_query = sanitize_query(query)

        response: Dict[str, Any] = {
            "lexical": [], "semantic": [], "reranked": [],
            "generation": {}, "mode": "fast", "error": None,
        }

        # ── BM25 Lexical Search (instant) ──
        t0 = time.perf_counter()
        try:
            response["lexical"] = self.pipeline.lexical_search(
                query=clean_query, top_k=lexical_limit,
                company=company, subfolder=subfolder,
            )
        except Exception as exc:
            print(f"[HybridRetriever] BM25 error: {exc}")
            response["lexical"] = []
        print(f"[Timer] BM25 search: {time.perf_counter() - t0:.3f}s")

        # Use BM25 results directly (fast, no cloud API needed)
        response["reranked"] = response["lexical"][:rerank_top_n]

        # Tenant safety validation
        response["lexical"]  = validate_context_tenant(response["lexical"],  company)
        response["reranked"] = validate_context_tenant(response["reranked"], company)

        # Select target chunks
        target_chunks = response["reranked"] or response["lexical"]

        # ── LLM Answer Generation ──
        t4 = time.perf_counter()
        response["generation"] = self.generator.generate_answer(
            query=clean_query,
            chunks=target_chunks,
            company=company,
            max_providers=max_providers,
            hf_model=hf_model,
            rag_mode=rag_mode,
            semantic_chunks=[],
            lexical_chunks=response["lexical"],
            use_local_engine=use_local_engine,
        )
        print(f"[Timer] LLM generation: {time.perf_counter() - t4:.3f}s")

        return response

"""
retriever.py
HybridRetriever — clean orchestration of the full LOVA HR search pipeline.

Steps:
  1. BM25 Lexical Search         (always available)
  2. Firestore Semantic Search   (requires Firebase)
  3. RRF Hybrid Fusion
  4. BGE Reranker V2 M3
  5. LLM Answer Generation (local Qwen2.5 / Cloud HF / Rule-based fallback)
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
    Falls back to BM25-only when Firebase is unreachable.
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
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """
        Execute the full retrieval + answer generation pipeline.

        Returns:
            {
                "lexical":    [...],   # BM25 top results
                "semantic":   [...],   # Firestore vector results
                "reranked":   [...],   # RRF + BGE Reranker results
                "generation": {...},   # LLM answer dict
                "mode":       str,     # "hybrid" | "bm25_only"
                "error":      str | None,
            }
        """
        # Sanitize query before searching
        clean_query = sanitize_query(query)

        response: Dict[str, Any] = {
            "lexical": [], "semantic": [], "reranked": [],
            "generation": {}, "mode": "hybrid", "error": None,
        }

        firebase_up = self._check_firebase()

        # Branch B: BM25 Lexical (always runs)
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

        # Branch A: Firebase Semantic + RRF + Rerank
        if firebase_up:
            try:
                # Use FAISS for fast semantic search if available
                t1 = time.perf_counter()
                if self.faiss_index and self.faiss_index.is_loaded:
                    # Compute query embedding
                    query_emb = self.pipeline._store._compute_query_embedding(clean_query)
                    response["semantic"] = self.faiss_index.search(
                        query_vector=query_emb, top_k=semantic_limit,
                        company=company, subfolder=subfolder
                    )
                    print(f"[HybridRetriever] FAISS semantic search: {len(response['semantic'])} results")
                else:
                    response["semantic"] = self.pipeline.semantic_search(
                        query=clean_query, top_k=semantic_limit,
                        company=company, subfolder=subfolder,
                    )
                print(f"[Timer] Semantic search: {time.perf_counter() - t1:.3f}s")
                
                # Fuse lexical + semantic
                t2 = time.perf_counter()
                lex_results = response["lexical"]
                sem_results = response["semantic"]
                from .embeddings import _reciprocal_rank_fusion
                fused = _reciprocal_rank_fusion([lex_results, sem_results])[:20]
                print(f"[Timer] RRF fusion: {time.perf_counter() - t2:.3f}s")
                
                # Skip reranking in fast mode for speed
                t3 = time.perf_counter()
                if rag_mode == "fast":
                    response["reranked"] = fused[:rerank_top_n]
                    print("[HybridRetriever] Fast mode: skipped reranking")
                else:
                    response["reranked"] = self.pipeline.rerank(
                        query=clean_query, candidates=fused, top_n=rerank_top_n,
                    )
                print(f"[Timer] Rerank: {time.perf_counter() - t3:.3f}s")
            except Exception as exc:
                print(f"[HybridRetriever] Firebase retrieval error: {exc}")
                response["error"] = f"Firebase retrieval error: {exc}"
                response["mode"] = "bm25_only"
                response["reranked"] = response["lexical"][:rerank_top_n]
        else:
            response["mode"] = "bm25_only"
            response["error"] = (
                "Firebase is not configured. Showing BM25-only results. "
                "Add serviceAccountKey.json to enable semantic search."
            )
            response["reranked"] = response["lexical"][:rerank_top_n]

        # Tenant safety validation
        response["lexical"]  = validate_context_tenant(response["lexical"],  company)
        response["semantic"] = validate_context_tenant(response["semantic"], company)
        response["reranked"] = validate_context_tenant(response["reranked"], company)

        # Select target chunks per rag_mode
        if rag_mode == "fullylexical":
            target_chunks = response["lexical"]
        elif rag_mode == "semantic":
            target_chunks = response["semantic"] or response["lexical"]
        else:
            target_chunks = response["reranked"] or response["lexical"]

        # LLM Answer Generation
        t4 = time.perf_counter()
        response["generation"] = self.generator.generate_answer(
            query=clean_query,
            chunks=target_chunks,
            company=company,
            max_providers=max_providers,
            hf_model=hf_model,
            rag_mode=rag_mode,
            semantic_chunks=response["semantic"],
            lexical_chunks=response["lexical"],
            use_local_engine=use_local_engine,
            conversation_context=conversation_context,
        )
        print(f"[Timer] LLM generation: {time.perf_counter() - t4:.3f}s")

        return response

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
        hf_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        use_local_engine: bool = True,
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
        try:
            response["lexical"] = self.pipeline.lexical_search(
                query=clean_query, top_k=lexical_limit,
                company=company, subfolder=subfolder,
            )
        except Exception as exc:
            print(f"[HybridRetriever] BM25 error: {exc}")
            response["lexical"] = []

        # Branch A: Firebase Semantic + RRF + Rerank
        if firebase_up:
            try:
                response["semantic"] = self.pipeline.semantic_search(
                    query=clean_query, top_k=semantic_limit,
                    company=company, subfolder=subfolder,
                )
                fused = self.pipeline.hybrid_search(
                    query=clean_query,
                    semantic_limit=semantic_limit, lexical_limit=lexical_limit,
                    fusion_top_k=20, company=company, subfolder=subfolder,
                )
                response["reranked"] = self.pipeline.rerank(
                    query=clean_query, candidates=fused, top_n=rerank_top_n,
                )
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
        )

        return response

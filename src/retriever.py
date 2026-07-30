"""
retriever.py
HybridRetriever — clean orchestration layer for the LOVA_HR search pipeline.

Used by app.py to execute queries. Provides:
  - search()  → {lexical, semantic, reranked} result dict
  - Graceful offline / Firebase connection-error fallback to BM25-only mode
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from .embeddings import EmbeddingPipeline
    from .config import is_firebase_available
    from .generator import LLMGenerator
except ImportError:
    from src.embeddings import EmbeddingPipeline  # type: ignore[no-redef]
    from src.config import is_firebase_available  # type: ignore[no-redef]
    from src.generator import LLMGenerator  # type: ignore[no-redef]


def validate_context_tenant(
    chunks: List[Dict[str, Any]],
    target_company: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Safety net validator: Inspects metadata right before returning to UI/LLM.
    Hard-blocks any chunk whose company does not match target_company when scoped search is active.
    """
    if not target_company or target_company == "All Companies":
        return chunks

    valid_chunks = []
    for chunk in chunks:
        chunk_company = chunk.get("company", chunk.get("company_id", ""))
        if chunk_company and chunk_company != target_company:
            print(
                f"[HybridRetriever] ⚠️ LEAK PREVENTED: Blocked chunk belonging to '{chunk_company}' "
                f"(target was '{target_company}')"
            )
            continue
        valid_chunks.append(chunk)

    return valid_chunks


class HybridRetriever:
    """
    Orchestrates the full retrieval pipeline for a user query:

    1. BM25 Lexical Search         (always available — no network required)
    2. Firestore Semantic Search   (requires Firebase connection)
    3. RRF Hybrid Fusion           (merges 1 + 2)
    4. BGE Reranker V2 M3          (cross-encoder scoring on fused top-15)
    5. LLM Free-Tier Generator     (Groq API / HF Serverless / Extractive Fallback)

    Falls back to BM25-only mode when Firebase is unreachable.
    """

    def __init__(self, pipeline: Optional[EmbeddingPipeline] = None):
        self.pipeline = pipeline or EmbeddingPipeline()
        self.generator = LLMGenerator()
        self._firebase_ok: Optional[bool] = None  # cached connection status

    # ── Firebase status (checked once per app session) ────────────────────────

    def _check_firebase(self) -> bool:
        if self._firebase_ok is None:
            self._firebase_ok = is_firebase_available()
            if not self._firebase_ok:
                print(
                    "[HybridRetriever] [Warning] Firebase unavailable — "
                    "falling back to in-memory mode."
                )
        return self._firebase_ok

    def reset_firebase_status(self) -> None:
        """Force re-check on next search (useful after credentials are updated)."""
        self._firebase_ok = None

    # ── Main search interface ─────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 3,
        company: Optional[str] = None,
        subfolder: Optional[str] = None,
        rerank_top_n: int = 5,
        semantic_limit: int = 20,
        lexical_limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Execute the full hybrid retrieval & answer generation pipeline for a query.

        Returns:
        {
            "lexical":    [...],  # BM25 results (top_k)
            "semantic":   [...],  # Firestore vector results (top_k)
            "reranked":   [...],  # RRF fused + BGE Reranker top results
            "generation": {...},  # LLM answer, citations & provider
            "mode":       "hybrid" | "bm25_only",
            "error":      None | str,  # error message if degraded
        }
        """
        response: Dict[str, Any] = {
            "lexical": [],
            "semantic": [],
            "reranked": [],
            "generation": {},
            "mode": "hybrid",
            "error": None,
        }

        firebase_up = self._check_firebase()

        # ── Branch B: BM25 Lexical (always runs) ─────────────────────────────
        try:
            response["lexical"] = self.pipeline.lexical_search(
                query=query,
                top_k=top_k,
                company=company,
                subfolder=subfolder,
            )
        except Exception as exc:
            print(f"[HybridRetriever] BM25 error: {exc}")
            response["lexical"] = []

        # ── Branch A + Fusion + Rerank (Firebase-dependent) ──────────────────
        if firebase_up:
            try:
                # Semantic search (display-only, top_k results)
                response["semantic"] = self.pipeline.semantic_search(
                    query=query,
                    top_k=top_k,
                    company=company,
                    subfolder=subfolder,
                )

                # Hybrid fusion (RRF over 20 + 20 candidates → top 15)
                fused = self.pipeline.hybrid_search(
                    query=query,
                    semantic_limit=semantic_limit,
                    lexical_limit=lexical_limit,
                    fusion_top_k=15,
                    company=company,
                    subfolder=subfolder,
                )

                # Rerank top-15 → top-5
                response["reranked"] = self.pipeline.rerank(
                    query=query,
                    candidates=fused,
                    top_n=rerank_top_n,
                )

            except Exception as exc:
                error_msg = f"Firebase retrieval error: {exc}"
                print(f"[HybridRetriever] {error_msg}")
                response["error"] = error_msg
                response["mode"] = "bm25_only"
                # Fallback: use BM25 as reranked results too
                response["reranked"] = response["lexical"][:rerank_top_n]
        else:
            response["mode"] = "bm25_only"
            response["error"] = (
                "Firebase is not configured or unreachable. "
                "Showing BM25-only results. "
                "Set up serviceAccountKey.json to enable semantic search."
            )
            # Still populate reranked with BM25 results so the tab has content
            response["reranked"] = response["lexical"][:rerank_top_n]

        # ── Post-retrieval safety net validation ─────────────────────────────
        response["lexical"] = validate_context_tenant(response["lexical"], company)
        response["semantic"] = validate_context_tenant(response["semantic"], company)
        response["reranked"] = validate_context_tenant(response["reranked"], company)

        # ── Free-Tier LLM Answer Generation ──────────────────────────────────
        response["generation"] = self.generator.generate_answer(
            query=query,
            chunks=response["reranked"],
            company=company,
        )

        return response

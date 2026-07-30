"""
embeddings.py
EmbeddingPipeline — chunking, Firestore sync, BM25 lexical search,
Firestore semantic search, RRF hybrid fusion, and BGE Reranker V2 M3 reranking.

Architecture:
    BM25 (lexical, Branch B) + Firestore find_nearest (semantic, Branch A)
        → RRF Fusion → Top-15
        → BGE Reranker V2 M3 → Top-5
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

try:
    from .firebase_client import FirestoreVectorStore
    from .config import BGE_RERANKER_MODEL
except ImportError:
    from src.firebase_client import FirestoreVectorStore  # type: ignore[no-redef]
    from src.config import BGE_RERANKER_MODEL  # type: ignore[no-redef]


# ── RRF constant (standard value from the RRF paper) ─────────────────────────
_RRF_K = 60


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.
        RRF_Score(doc) = Σ 1 / (k + rank_i(doc))

    Args:
        ranked_lists: Each inner list is a ranked list of result dicts
                      (must have a 'doc_id' key).
    Returns:
        A single merged, deduplicated list sorted by descending RRF score.
        Each result dict gains an 'rrf_score' key.
    """
    scores: Dict[str, float] = {}
    doc_data: Dict[str, Dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = doc.get("doc_id") or doc.get("chunk_text", "")[:64]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank)
            if doc_id not in doc_data:
                doc_data[doc_id] = doc.copy()

    merged = []
    for doc_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = doc_data[doc_id].copy()
        entry["rrf_score"] = rrf_score
        merged.append(entry)

    return merged


def is_junk_chunk(text: str) -> bool:
    """
    Returns True if the chunk is boilerplate noise like a Table of Contents,
    index page, or cover page.
    """
    if not text or not text.strip():
        return True

    text_lower = text.lower()

    # Check 1: Explicit TOC headers
    if "table of contents" in text_lower or "index page" in text_lower or "table of content" in text_lower:
        return True

    # Check 2: High density of page numbers / dot leaders (e.g. "OBJECTIVE ........ 5")
    dot_leader_pattern = r"\.{3,}\s*\d+"
    if len(re.findall(dot_leader_pattern, text)) > 2:
        return True

    # Check 3: High ratio of index lines ending with page numbers
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        toc_lines = [line for line in lines if re.search(r"\.{2,}\s*\d+$|\bpage\s+\d+$|\b\d+$", line, re.IGNORECASE)]
        if len(lines) >= 3 and (len(toc_lines) / len(lines)) > 0.4:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
class EmbeddingPipeline:
    """
    Orchestrates the full LOVA_HR retrieval pipeline:
      1. chunk_documents()     — split LangChain docs into chunks
      2. sync_vector_store()   — upsert chunks into Firestore
      3. lexical_search()      — BM25 on in-memory corpus (from Firestore)
      4. semantic_search()     — Firestore native vector search
      5. hybrid_search()       — RRF fusion of BM25 + semantic
      6. rerank()              — BGE Reranker V2 M3 cross-encoder on top-15
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = FirestoreVectorStore()

        # BM25 cache: (company, subfolder) → (BM25Okapi index, list of doc dicts)
        self._bm25_cache: Dict[Tuple[str, str], Tuple[BM25Okapi, List[Dict[str, Any]]]] = {}

        # Reranker (lazy-loaded on first call to rerank())
        self._reranker = None

    # ── Chunking ──────────────────────────────────────────────────────────────
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        clean_chunks = [c for c in chunks if not is_junk_chunk(c.page_content)]
        print(f"[Pipeline] {len(documents)} docs → {len(chunks)} chunks ({len(chunks) - len(clean_chunks)} TOC/junk chunks stripped)")
        return clean_chunks

    # ── Vector store sync ─────────────────────────────────────────────────────
    def sync_vector_store(self, chunks: List[Any], force_reload: bool = False) -> None:
        """Upsert chunks into Firestore (idempotent via content-hash IDs)."""
        # Invalidate BM25 cache so new docs are picked up on next search
        self.invalidate_bm25_cache()
        self.vector_store.upload_policy_chunks(chunks, force_reload=force_reload)

    def invalidate_bm25_cache(self) -> None:
        """Clear cached BM25 indexes so deleted or modified files are refreshed."""
        self._bm25_cache.clear()

    # ── BM25 cache helpers ────────────────────────────────────────────────────
    def _get_bm25_index(
        self,
        company: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> Tuple[Optional[BM25Okapi], List[Dict[str, Any]]]:
        """
        Return (BM25Okapi index, corpus docs) for the given scope.
        Hydrates from Firestore on cache miss; returns (None, []) on error.
        """
        cache_key = (company or "ALL", subfolder or "ALL")
        if cache_key in self._bm25_cache:
            return self._bm25_cache[cache_key]

        print(f"[Pipeline] Building BM25 index for scope {cache_key} from Firestore...")
        try:
            docs = self.vector_store.get_all_documents(
                company=company, subfolder=subfolder
            )
        except Exception as exc:
            print(f"[Pipeline] BM25 Firestore fetch error: {exc}")
            docs = []

        if not docs:
            return None, []

        corpus = [_tokenize(d["chunk_text"]) for d in docs]
        index = BM25Okapi(corpus)

        # Clip negative IDF values to prevent negative scores in small corpora
        if hasattr(index, "idf"):
            index.idf = {k: max(v, 0.0001) for k, v in index.idf.items()}

        self._bm25_cache[cache_key] = (index, docs)
        print(f"[Pipeline] BM25 index built ({len(docs)} documents).")
        return index, docs

    def invalidate_bm25_cache(self) -> None:
        """Call this after uploading new documents to force BM25 index rebuild."""
        self._bm25_cache.clear()

    # ── Lexical Search (BM25) ─────────────────────────────────────────────────
    def lexical_search(
        self,
        query: str,
        chunks: Optional[List[Any]] = None,  # kept for API compatibility
        top_k: int = 3,
        company: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run BM25 search. The corpus is fetched from Firestore and cached in memory.
        The `chunks` parameter is accepted for backward compatibility but ignored —
        the BM25 corpus is always sourced from the live Firestore collection.
        """
        bm25, docs = self._get_bm25_index(company=company, subfolder=subfolder)
        if bm25 is None or not docs:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            doc = docs[idx]
            doc_company = doc.get("company", doc.get("company_id", "General"))
            if company and company != "All Companies" and doc_company != company:
                continue
            results.append(
                {
                    "doc_id": doc["doc_id"],
                    "chunk_text": doc["chunk_text"],
                    "source_file": doc["source_file"],
                    "company": doc_company,
                    "company_id": doc_company,
                    "subfolder": doc["subfolder"],
                    "filename": doc["filename"],
                    "score": score,
                }
            )
        return results

    # ── Semantic Search (Firestore Dense Vector) ──────────────────────────────
    def semantic_search(
        self,
        query: str,
        top_k: int = 3,
        company: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query Firestore with a BGE-M3 dense vector and optional scoping.
        Returns top-k cosine-similarity results.
        """
        try:
            results = self.vector_store.vector_search_firestore(
                query=query,
                limit=top_k,
                company=company,
                subfolder=subfolder,
            )
        except Exception as exc:
            print(f"[Pipeline] Firestore semantic search error: {exc}")
            results = []
        return results

    # ── Hybrid Search (BM25 + Firestore → RRF) ───────────────────────────────
    def hybrid_search(
        self,
        query: str,
        semantic_limit: int = 20,
        lexical_limit: int = 20,
        fusion_top_k: int = 15,
        company: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search via Reciprocal Rank Fusion (RRF):
          Branch A: Firestore vector search (semantic, BGE-M3)
          Branch B: BM25 Okapi (lexical, from Firestore corpus)
          Fusion:   RRF score = Σ 1 / (60 + rank_i)

        Returns top `fusion_top_k` deduplicated results with 'rrf_score'.
        """
        # Branch A — Semantic (Firestore)
        semantic_results: List[Dict[str, Any]] = []
        try:
            semantic_results = self.vector_store.vector_search_firestore(
                query=query,
                limit=semantic_limit,
                company=company,
                subfolder=subfolder,
            )
        except Exception as exc:
            print(f"[Pipeline] Hybrid: semantic branch error — {exc}")

        # Branch B — Lexical (BM25)
        lexical_results: List[Dict[str, Any]] = []
        try:
            lexical_results = self.lexical_search(
                query=query,
                top_k=lexical_limit,
                company=company,
                subfolder=subfolder,
            )
        except Exception as exc:
            print(f"[Pipeline] Hybrid: lexical branch error — {exc}")

        if not semantic_results and not lexical_results:
            return []

        fused = _reciprocal_rank_fusion([semantic_results, lexical_results])
        return fused[:fusion_top_k]

    # ── Reranking (BGE Reranker V2 M3) ───────────────────────────────────────
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Score `candidates` against `query` using BGE Reranker V2 M3 (cross-encoder).
        Returns the top_n results sorted by descending reranker score.

        Each result gains a 'reranker_score' key (raw logit, higher = more relevant).
        """
        if not candidates:
            return []

        if self._reranker is None:
            print(f"[Pipeline] Loading reranker '{BGE_RERANKER_MODEL}'...")
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(BGE_RERANKER_MODEL)
            print("[Pipeline] Reranker loaded.")

        pairs = [(query, c["chunk_text"]) for c in candidates]
        try:
            scores = self._reranker.predict(pairs)
            scores = np.atleast_1d(scores)
        except Exception as exc:
            print(f"[Pipeline] Reranker predict error: {exc}")
            # Fall back: return candidates ordered by rrf_score
            return sorted(
                candidates,
                key=lambda x: x.get("rrf_score", x.get("score", 0)),
                reverse=True,
            )[:top_n]

        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for reranker_score, doc in scored[:top_n]:
            entry = doc.copy()
            entry["reranker_score"] = float(reranker_score)
            results.append(entry)


        return results

"""
embeddings.py
EmbeddingPipeline: chunking, Firestore sync, BM25 lexical search,
Firestore semantic search, RRF hybrid fusion, BGE Reranker V2 M3.

Architecture:
    BM25 (lexical) + Firestore find_nearest (semantic)
        -> RRF Fusion -> Top-20
        -> BGE Reranker V2 M3 -> Top-5
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

try:
    from .firebase_client import FirestoreVectorStore
    from .config import BGE_RERANKER_MODEL
    from .preprocessor import sanitize_chunk, is_junk_chunk
except ImportError:
    from src.firebase_client import FirestoreVectorStore
    from src.config import BGE_RERANKER_MODEL
    from src.preprocessor import sanitize_chunk, is_junk_chunk

_RRF_K = 60


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _reciprocal_rank_fusion(ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge multiple ranked lists via Reciprocal Rank Fusion."""
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


class EmbeddingPipeline:
    """
    Full retrieval pipeline:
      - In-memory BGE embeddings (BAAI/bge-large-en-v1.5) + BM25 index built from loaded documents
      - Firestore semantic search (if Firebase available)
      - RRF fusion + BGE Reranker V2 M3
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._store = FirestoreVectorStore()
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Dict[str, Any]] = []
        self._reranker = None
        self._reranker_loaded = False

    @property
    def vector_store(self) -> FirestoreVectorStore:
        return self._store

    def chunk_documents(self, langchain_docs: List[Any]) -> List[Any]:
        """Chunk LangChain Document objects and return filtered Document list."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        from langchain_core.documents import Document
        results = []
        for doc in langchain_docs:
            splits = splitter.split_text(doc.page_content)
            for split_text in splits:
                clean_text = sanitize_chunk(split_text)
                if clean_text and not is_junk_chunk(clean_text):
                    results.append(Document(page_content=clean_text, metadata=doc.metadata.copy()))
        return results

    def sync_vector_store(self, chunks: List[Any]) -> None:
        """Upload chunks to vector store and build BM25 index."""
        self.add_documents(chunks)

    # ── Document index ────────────────────────────────────────────────────────

    def build_bm25_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build an in-memory BM25 index from sanitized chunks."""
        self._bm25_docs = chunks
        tokenized = [_tokenize(c.get("chunk_text", "")) for c in chunks]
        if any(tokenized):
            self._bm25 = BM25Okapi(tokenized)
        print(f"[EmbeddingPipeline] BM25 index built: {len(chunks)} chunks")

    def add_documents(self, langchain_docs: List[Any]) -> None:
        """Chunk LangChain Documents, sanitize, embed, and upsert to Firestore."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks_to_index: List[Dict[str, Any]] = []
        for doc in langchain_docs:
            splits = splitter.split_text(doc.page_content if hasattr(doc, "page_content") else str(doc))
            for i, split_text in enumerate(splits):
                clean_text = sanitize_chunk(split_text)
                if not clean_text or is_junk_chunk(clean_text):
                    continue
                meta = doc.metadata.copy() if hasattr(doc, "metadata") else {}
                chunk_dict = {**meta, "chunk_text": clean_text,
                              "chunk_index": i, "doc_id": f"{meta.get('filename','doc')}_{meta.get('page',1)}_{i}"}
                chunks_to_index.append(chunk_dict)

        if chunks_to_index:
            try:
                from .config import is_firebase_available
                if is_firebase_available():
                    self._store.upsert_chunks(chunks_to_index)
            except Exception as exc:
                print(f"[EmbeddingPipeline] Firestore upsert skipped: {exc}")
            self.build_bm25_index(chunks_to_index)
            print(f"[EmbeddingPipeline] Indexed {len(chunks_to_index)} chunks")

    # ── Search methods ────────────────────────────────────────────────────────

    def lexical_search(
        self, query: str, top_k: int = 10,
        company: Optional[str] = None, subfolder: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """BM25 search over the in-memory index."""
        if self._bm25 is None or not self._bm25_docs:
            # Fall back to Firestore-stored chunks loaded on demand
            try:
                all_chunks = self._store.get_all_chunks(company=company, subfolder=subfolder)
                if all_chunks:
                    self.build_bm25_index(all_chunks)
            except Exception:
                return []
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in indices:
            if scores[idx] <= 0:
                continue
            doc = self._bm25_docs[idx].copy()
            doc["bm25_score"] = float(scores[idx])
            if company and doc.get("company", doc.get("company_id")) != company:
                continue
            results.append(doc)
        return results

    def semantic_search(
        self, query: str, top_k: int = 10,
        company: Optional[str] = None, subfolder: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Firestore vector similarity search."""
        if hasattr(self._store, "search"):
            return self._store.search(query=query, top_k=top_k, company=company, subfolder=subfolder)
        elif hasattr(self._store, "vector_search_firestore"):
            return self._store.vector_search_firestore(query=query, limit=top_k, company=company, subfolder=subfolder)
        return []

    def hybrid_search(
        self, query: str,
        semantic_limit: int = 20, lexical_limit: int = 20, fusion_top_k: int = 20,
        company: Optional[str] = None, subfolder: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """RRF fusion of lexical + semantic results."""
        lex = self.lexical_search(query, top_k=lexical_limit, company=company, subfolder=subfolder)
        sem = self.semantic_search(query, top_k=semantic_limit, company=company, subfolder=subfolder)
        fused = _reciprocal_rank_fusion([lex, sem])
        return fused[:fusion_top_k]

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """BGE Reranker V2 M3 cross-encoder scoring."""
        if not candidates:
            return []
        if not self._reranker_loaded:
            try:
                from FlagEmbedding import FlagReranker
                print(f"[EmbeddingPipeline] Loading reranker '{BGE_RERANKER_MODEL}'...")
                self._reranker = FlagReranker(BGE_RERANKER_MODEL, use_fp16=False)
                self._reranker_loaded = True
                print("[EmbeddingPipeline] Reranker loaded.")
            except Exception as exc:
                print(f"[EmbeddingPipeline] Reranker load failed: {exc}. Using RRF score ordering.")
                self._reranker_loaded = True
                self._reranker = None

        if self._reranker is None:
            return sorted(candidates, key=lambda d: d.get("rrf_score", 0), reverse=True)[:top_n]

        pairs = [[query, c.get("chunk_text", "")] for c in candidates]
        try:
            if self._reranker is not None:
                scores = self._reranker.compute_score(pairs)
                if not isinstance(scores, list):
                    scores = [scores]
                for doc, score in zip(candidates, scores):
                    doc["reranker_score"] = float(score)
                return sorted(candidates, key=lambda d: d.get("reranker_score", 0), reverse=True)[:top_n]
        except Exception as exc:
            print(f"[EmbeddingPipeline] FlagReranker error ({exc}) — trying CrossEncoder fallback.")
            self._reranker = None

        # Fallback to SentenceTransformer CrossEncoder
        try:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(BGE_RERANKER_MODEL)
            ce_scores = ce.predict(pairs)
            for doc, score in zip(candidates, ce_scores):
                doc["reranker_score"] = float(score)
            return sorted(candidates, key=lambda d: d.get("reranker_score", 0), reverse=True)[:top_n]
        except Exception as exc:
            print(f"[EmbeddingPipeline] CrossEncoder fallback warning ({exc}) — using RRF ordering.")
            return sorted(candidates, key=lambda d: d.get("rrf_score", 0), reverse=True)[:top_n]

"""
firebase_client.py
FirestoreVectorStore — Firestore-backed dense vector store using BAAI/bge-m3.

This module replaces ChromaVectorStore as the primary data layer.

Firestore document schema (collection: hr_policies):
{
    "doc_id":    "<sha256-content-hash>",
    "title":     "<filename>",
    "content":   "<chunk text>",
    "embedding": FieldValue.vector([...1024 floats...]),
    "keywords":  ["token1", "token2", ...],   # for BM25 hydration
    "metadata":  {
        "company":   "TCS",
        "subfolder": "Policies",
        "filename":  "tcs_hr_policies.pdf",
        "source":    "/absolute/path/to/file",
        "page":      3
    }
}
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

try:
    from .config import (
        COLLECTION_NAME,
        BGE_EMBEDDING_MODEL,
        EMBEDDING_DIM,
        get_firestore_client,
        init_firebase,
        is_firebase_available,
    )
except ImportError:
    from src.config import (  # type: ignore[no-redef]
        COLLECTION_NAME,
        BGE_EMBEDDING_MODEL,
        EMBEDDING_DIM,
        get_firestore_client,
        init_firebase,
        is_firebase_available,
    )


# ── Tokenizer helper (shared with BM25) ──────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


# ── Content-hash ID ──────────────────────────────────────────────────────────
def _chunk_id(text: str, metadata: Dict[str, Any], index: int = 0) -> str:
    source = str(metadata.get("source", ""))
    company = str(metadata.get("company", ""))
    subfolder = str(metadata.get("subfolder", ""))
    page = str(metadata.get("page", ""))
    raw = f"{source}:{company}:{subfolder}:{page}:{index}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ─────────────────────────────────────────────────────────────────────────────
class FirestoreVectorStore:
    """
    Firestore-backed vector store using BAAI/bge-m3 1024-dimensional embeddings.
    Includes in-memory fallback when Firebase credentials are unavailable.

    Provides:
        - upload_policy_chunks()   — batch upsert LangChain Document chunks
        - vector_search_firestore()— Firestore native cosine vector search (with in-memory fallback)
        - get_all_documents()      — stream all docs for BM25 index hydration
        - delete_*()               — scoped deletion helpers
        - clear()                  — wipe collection
    """

    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        self._model = None          # lazy-loaded BGE-M3 embedding model
        self._db = None             # lazy-loaded Firestore client
        self._memory_store: Dict[str, Dict[str, Any]] = {}  # in-memory store fallback
        print(f"[FirestoreVectorStore] Initialized (model will load on first use).")

    # ── Lazy loaders & Connectivity ──────────────────────────────────────────

    @property
    def is_online(self) -> bool:
        """Check if Firestore client is initialized and accessible."""
        if self._db is not None:
            return True
        if is_firebase_available():
            try:
                self._db = get_firestore_client()
                return True
            except Exception:
                return False
        return False

    @property
    def model(self):
        """Lazy-load BAAI/bge-m3 using sentence-transformers for clean API consistency."""
        if self._model is None:
            print(f"[FirestoreVectorStore] Loading embedding model '{BGE_EMBEDDING_MODEL}'...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(BGE_EMBEDDING_MODEL)
            print(f"[FirestoreVectorStore] Model loaded (dim={EMBEDDING_DIM}).")
        return self._model

    @property
    def db(self):
        """Lazy-load authenticated Firestore client."""
        if self._db is None:
            self._db = get_firestore_client()
        return self._db

    @property
    def collection(self):
        if not self.is_online:
            raise EnvironmentError("Firestore is unavailable.")
        return self.db.collection(COLLECTION_NAME)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Return a list of 1024-dim float lists using BGE-M3."""
        vecs = self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = vecs / norms
        return vecs.tolist()

    def _compute_query_embedding(self, query: str) -> List[float]:
        """Encode a single query string using BGE-M3."""
        vec = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    # ── Write ─────────────────────────────────────────────────────────────────

    def upload_policy_chunks(
        self,
        chunks: List[Any],
        force_reload: bool = False,
    ) -> None:
        """
        Batch-upsert LangChain Document chunks into Firestore and in-memory fallback.
        """
        if force_reload:
            self.clear()

        if not chunks:
            print("[FirestoreVectorStore] No chunks to upload.")
            return

        # Build document list
        docs_to_upsert: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            meta = getattr(chunk, "metadata", {})
            doc_id = _chunk_id(chunk.page_content, meta, i)
            docs_to_upsert.append(
                {
                    "doc_id": doc_id,
                    "title": str(meta.get("filename", f"doc_{i}")),
                    "content": chunk.page_content,
                    "keywords": _tokenize(chunk.page_content)[:200],  # cap for Firestore limits
                    "metadata": {
                        "company": str(meta.get("company", meta.get("company_id", "General"))),
                        "company_id": str(meta.get("company_id", meta.get("company", "General"))),
                        "subfolder": str(meta.get("subfolder", "General")),
                        "filename": str(meta.get("filename", meta.get("document_id", ""))),
                        "document_id": str(meta.get("document_id", meta.get("filename", ""))),
                        "source": str(meta.get("source", "")),
                        "page": int(meta.get("page", meta.get("page_number", 1))),
                        "page_number": int(meta.get("page_number", meta.get("page", 1))),
                        "chunk_hash": doc_id,
                    },
                }
            )

        # Filter out already-stored doc IDs
        existing_ids = self._get_existing_ids([d["doc_id"] for d in docs_to_upsert])
        new_docs = [d for d in docs_to_upsert if d["doc_id"] not in existing_ids]

        if not new_docs:
            print(f"[FirestoreVectorStore] All {len(chunks)} chunks already stored — skipping.")
            return

        print(
            f"[FirestoreVectorStore] Embedding {len(new_docs)} new chunks "
            f"(skipping {len(existing_ids)} existing)..."
        )

        # Embed in batches and store
        total_written = 0
        for batch_start in range(0, len(new_docs), self.batch_size):
            batch_docs = new_docs[batch_start : batch_start + self.batch_size]
            texts = [d["content"] for d in batch_docs]

            embeddings = self._compute_embeddings(texts)

            # Store in local in-memory store
            for doc_data, emb in zip(batch_docs, embeddings):
                self._memory_store[doc_data["doc_id"]] = {
                    **doc_data,
                    "embedding": emb,
                    "uploaded_at": int(time.time()),
                }

            # Write to Firestore if online
            if self.is_online:
                try:
                    fs_batch = self.db.batch()
                    for doc_data, emb in zip(batch_docs, embeddings):
                        ref = self.collection.document(doc_data["doc_id"])
                        fs_batch.set(
                            ref,
                            {
                                **doc_data,
                                "embedding": Vector(emb),
                                "uploaded_at": int(time.time()),
                            },
                            merge=True,
                        )
                    fs_batch.commit()
                except Exception as exc:
                    print(f"[FirestoreVectorStore] Firestore write notice (using in-memory): {exc}")

            total_written += len(batch_docs)
            print(
                f"[FirestoreVectorStore] Written batch {batch_start // self.batch_size + 1}: "
                f"{total_written}/{len(new_docs)} chunks."
            )

        print(f"[FirestoreVectorStore] Upload complete — {total_written} new chunks stored.")

    def _get_existing_ids(self, doc_ids: List[str]) -> set:
        """Check which doc IDs already exist in memory or Firestore."""
        existing: set = set(doc_id for doc_id in doc_ids if doc_id in self._memory_store)
        if self.is_online:
            for i in range(0, len(doc_ids), 30):
                batch_ids = [d for d in doc_ids[i : i + 30] if d not in existing]
                if not batch_ids:
                    continue
                try:
                    docs = self.collection.where("doc_id", "in", batch_ids).stream()
                    for d in docs:
                        existing.add(d.id)
                except Exception as exc:
                    print(f"[FirestoreVectorStore] Warning checking existing IDs: {exc}")
        return existing

    # ── Read: Vector Search ───────────────────────────────────────────────────

    def vector_search_firestore(
        self,
        query: str,
        limit: int = 20,
        company: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search via Firestore native vector index or in-memory BGE-M3 cosine search.
        """
        query_vector = self._compute_query_embedding(query)

        if self.is_online:
            try:
                col_ref = self.collection
                if company and company != "All Companies":
                    col_ref = col_ref.where("metadata.company", "==", company)
                    if subfolder and subfolder != "All Subfolders":
                        col_ref = col_ref.where("metadata.subfolder", "==", subfolder)

                vector_query = col_ref.find_nearest(
                    vector_field="embedding",
                    query_vector=Vector(query_vector),
                    distance_measure=DistanceMeasure.COSINE,
                    limit=limit * 2,
                )
                docs = vector_query.get()
                results = []
                for doc in docs:
                    data = doc.to_dict()
                    meta = data.get("metadata", {})
                    doc_company = meta.get("company", meta.get("company_id", "General"))
                    doc_subfolder = meta.get("subfolder", "General")
                    doc_id = data.get("doc_id", doc.id)
                    filename_val = meta.get("filename", meta.get("document_id", ""))
                    page_val = meta.get("page", meta.get("page_number", 1))

                    if company and company != "All Companies":
                        if doc_company != company:
                            continue
                        if subfolder and subfolder != "All Subfolders":
                            if doc_subfolder != subfolder:
                                continue

                    distance = data.get("distance", 0.0)
                    similarity = max(0.0, 1.0 - float(distance))
                    results.append(
                        {
                            "doc_id": doc_id,
                            "chunk_hash": doc_id,
                            "chunk_text": data.get("content", ""),
                            "source_file": meta.get("source", "—"),
                            "company": doc_company,
                            "company_id": doc_company,
                            "subfolder": doc_subfolder,
                            "filename": filename_val,
                            "document_id": filename_val,
                            "page": page_val,
                            "page_number": page_val,
                            "score": similarity,
                            "metadata": meta,
                        }
                    )
                    if len(results) >= limit:
                        break
                if results:
                    return results
            except Exception as exc:
                print(f"[FirestoreVectorStore] Firestore search notice (falling back to in-memory): {exc}")

        # In-memory vector search fallback
        q_vec = np.array(query_vector)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scored_docs = []
        for doc_data in self._memory_store.values():
            meta = doc_data.get("metadata", {})
            doc_company = meta.get("company", meta.get("company_id", "General"))
            doc_subfolder = meta.get("subfolder", "General")

            if company and company != "All Companies":
                if doc_company != company:
                    continue
                if subfolder and subfolder != "All Subfolders":
                    if doc_subfolder != subfolder:
                        continue

            emb = doc_data.get("embedding", [])
            if not emb:
                continue
            emb_vec = np.array(emb)
            emb_norm = np.linalg.norm(emb_vec)
            if emb_norm > 0:
                emb_vec = emb_vec / emb_norm
            similarity = float(np.dot(q_vec, emb_vec))

            scored_docs.append(
                {
                    "doc_id": doc_data["doc_id"],
                    "chunk_text": doc_data.get("content", ""),
                    "source_file": meta.get("source", "—"),
                    "company": doc_company,
                    "subfolder": doc_subfolder,
                    "filename": meta.get("filename", ""),
                    "score": max(0.0, similarity),
                }
            )

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:limit]

    # ── Read: All Documents (for BM25 hydration) ─────────────────────────────

    def get_all_documents(
        self,
        company: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Stream all documents from Firestore or in-memory store (optionally filtered).
        """
        if self.is_online:
            try:
                query = self.collection
                if company and company != "All Companies":
                    query = query.where("metadata.company", "==", company)
                    if subfolder and subfolder != "All Subfolders":
                        query = query.where("metadata.subfolder", "==", subfolder)
                docs = query.stream()
                results = []
                for doc in docs:
                    data = doc.to_dict()
                    meta = data.get("metadata", {})
                    doc_id = data.get("doc_id", doc.id)
                    company_val = meta.get("company", meta.get("company_id", "General"))
                    filename_val = meta.get("filename", meta.get("document_id", ""))
                    page_val = meta.get("page", meta.get("page_number", 1))
                    results.append(
                        {
                            "doc_id": doc_id,
                            "chunk_hash": doc_id,
                            "chunk_text": data.get("content", ""),
                            "keywords": data.get("keywords", []),
                            "source_file": meta.get("source", "—"),
                            "company": company_val,
                            "company_id": company_val,
                            "subfolder": meta.get("subfolder", "General"),
                            "filename": filename_val,
                            "document_id": filename_val,
                            "page": page_val,
                            "page_number": page_val,
                            "score": 0.0,
                            "metadata": meta,
                        }
                    )
                if results:
                    return results
            except Exception as exc:
                print(f"[FirestoreVectorStore] get_all_documents Firestore notice: {exc}")

        # In-memory fallback
        results = []
        for data in self._memory_store.values():
            meta = data.get("metadata", {})
            doc_company = meta.get("company", meta.get("company_id", "General"))
            doc_subfolder = meta.get("subfolder", "General")
            doc_id = data.get("doc_id", "")
            filename_val = meta.get("filename", meta.get("document_id", ""))
            page_val = meta.get("page", meta.get("page_number", 1))

            if company and company != "All Companies":
                if doc_company != company:
                    continue
                if subfolder and subfolder != "All Subfolders":
                    if doc_subfolder != subfolder:
                        continue

            results.append(
                {
                    "doc_id": doc_id,
                    "chunk_hash": doc_id,
                    "chunk_text": data.get("content", ""),
                    "keywords": data.get("keywords", []),
                    "source_file": meta.get("source", "—"),
                    "company": doc_company,
                    "company_id": doc_company,
                    "subfolder": doc_subfolder,
                    "filename": filename_val,
                    "document_id": filename_val,
                    "page": page_val,
                    "page_number": page_val,
                    "score": 0.0,
                    "metadata": meta,
                }
            )
        return results

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_company_chunks(self, company: str) -> None:
        """Delete all documents belonging to a company."""
        if not company or company == "All Companies":
            return
        to_del = [
            doc_id for doc_id, d in self._memory_store.items()
            if d.get("metadata", {}).get("company") == company
        ]
        for doc_id in to_del:
            del self._memory_store[doc_id]

        if self.is_online:
            try:
                self._batch_delete(self.collection.where("metadata.company", "==", company))
            except Exception as exc:
                print(f"[FirestoreVectorStore] delete_company_chunks error: {exc}")
        print(f"[FirestoreVectorStore] Deleted all chunks for company '{company}'.")

    def delete_subfolder_chunks(self, company: str, subfolder: str) -> None:
        """Delete all documents belonging to a company/subfolder."""
        if not company or not subfolder:
            return
        to_del = [
            doc_id for doc_id, d in self._memory_store.items()
            if d.get("metadata", {}).get("company") == company
            and d.get("metadata", {}).get("subfolder") == subfolder
        ]
        for doc_id in to_del:
            del self._memory_store[doc_id]

        if self.is_online:
            try:
                query = (
                    self.collection
                    .where("metadata.company", "==", company)
                    .where("metadata.subfolder", "==", subfolder)
                )
                self._batch_delete(query)
            except Exception as exc:
                print(f"[FirestoreVectorStore] delete_subfolder_chunks error: {exc}")
        print(f"[FirestoreVectorStore] Deleted chunks for {company}/{subfolder}.")

    def delete_file_chunks(self, source_path: str) -> None:
        """Delete all documents belonging to a specific source file."""
        from pathlib import Path as _Path
        try:
            source_str = str(_Path(source_path).resolve())
        except Exception:
            source_str = str(source_path)

        to_del = []
        for doc_id, d in self._memory_store.items():
            meta_src = d.get("metadata", {}).get("source", "")
            try:
                meta_src_norm = str(_Path(meta_src).resolve())
            except Exception:
                meta_src_norm = str(meta_src)
            if meta_src_norm == source_str or meta_src == source_path:
                to_del.append(doc_id)

        for doc_id in to_del:
            del self._memory_store[doc_id]

        if self.is_online:
            try:
                self._batch_delete(self.collection.where("metadata.source", "==", source_str))
            except Exception as exc:
                print(f"[FirestoreVectorStore] delete_file_chunks error: {exc}")
        print(f"[FirestoreVectorStore] Deleted chunks for source: {source_str}")

    def clear(self) -> None:
        """Delete ALL documents in the collection."""
        print("[FirestoreVectorStore] Clearing entire collection...")
        self._memory_store.clear()
        if self.is_online:
            try:
                self._batch_delete(self.collection)
            except Exception as exc:
                print(f"[FirestoreVectorStore] clear error: {exc}")
        print("[FirestoreVectorStore] Collection cleared.")

    def _batch_delete(self, query) -> None:
        """Delete all documents matching a query in Firestore batches."""
        batch_size = 400
        deleted = 0
        while True:
            docs = list(query.limit(batch_size).stream())
            if not docs:
                break
            fs_batch = self.db.batch()
            for doc in docs:
                fs_batch.delete(doc.reference)
            fs_batch.commit()
            deleted += len(docs)
            if len(docs) < batch_size:
                break
        print(f"[FirestoreVectorStore] Batch-deleted {deleted} documents.")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return document count."""
        if self.is_online:
            try:
                agg = self.collection.count()
                result = agg.get()
                return result[0][0].value
            except Exception:
                pass
        return len(self._memory_store)


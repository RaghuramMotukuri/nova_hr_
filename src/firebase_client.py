"""
firebase_client.py
FirestoreVectorStore — Firestore-backed dense vector store using BAAI/bge-large-en-v1.5.

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
        "subtopic":  "Maternity Leave",
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

# Lazy imports — defer heavy libraries until first use
np = None  # type: ignore
faiss = None  # type: ignore
_FieldFilter = None  # type: ignore
_DistanceMeasure = None  # type: ignore
_Vector = None  # type: ignore


def _ensure_numpy():
    """Lazy-import numpy and Firestore vector types on first use."""
    global np, _FieldFilter, _DistanceMeasure, _Vector
    if np is None:
        import numpy as _np
        from google.cloud.firestore_v1.base_query import FieldFilter as _FF
        from google.cloud.firestore_v1.base_vector_query import DistanceMeasure as _DM
        from google.cloud.firestore_v1.vector import Vector as _V
        np = _np
        _FieldFilter = _FF
        _DistanceMeasure = _DM
        _Vector = _V

try:
    from .config import (
        COLLECTION_NAME,
        BGE_EMBEDDING_MODEL,
        EMBEDDING_DIM,
        CLOUD_ONLY_MODE,
        get_firestore_client,
        init_firebase,
        is_firebase_available,
    )
except ImportError:
    from src.config import (  # type: ignore[no-redef]
        COLLECTION_NAME,
        BGE_EMBEDDING_MODEL,
        EMBEDDING_DIM,
        CLOUD_ONLY_MODE,
        get_firestore_client,
        init_firebase,
        is_firebase_available,
    )


def _safe_int(val: Any, default: int = 1) -> int:
    if val is None or val == "":
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


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
    _has_native_vector_index: Optional[bool] = None  # None=unknown, True=online, False=use in-memory engine
    _printed_index_notice: bool = False

    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        self._model = None          # lazy-loaded BGE embedding model
        self._db = None             # lazy-loaded Firestore client
        self._memory_store: Dict[str, Dict[str, Any]] = {}  # in-memory store fallback
        print(f"[FirestoreVectorStore] Initialized (model will load on first use).")

    def upsert_chunks(self, chunks: List[Any], force_reload: bool = False) -> None:
        """Alias for upload_policy_chunks — accepts list of dicts or Document objects."""
        if not chunks:
            return
        # Convert dicts with 'chunk_text' into objects compatible with upload_policy_chunks
        processed = []
        for c in chunks:
            if isinstance(c, dict):
                text = c.get("chunk_text") or c.get("content", "")
                doc_id = c.get("doc_id", "")
                meta = {k: v for k, v in c.items() if k not in ("chunk_text", "content")}
                processed.append({"page_content": text, "metadata": meta, "doc_id": doc_id})
            else:
                processed.append(c)
        self.upload_policy_chunks(processed, force_reload=force_reload)

    def get_all_chunks(self, company: Optional[str] = None, subfolder: Optional[str] = None) -> List[Dict[str, Any]]:
        """Alias for get_all_documents — used by EmbeddingPipeline for BM25 hydration."""
        return self.get_all_documents(company=company, subfolder=subfolder)

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
        """Lazy-load the configured BGE embedding model (default BAAI/bge-large-en-v1.5)."""
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

    def _call_hf_cloud_embedding(self, text: str) -> List[float]:
        """Compute 1024-dim dense embedding directly via HuggingFace Cloud Feature Extraction API."""
        _ensure_numpy()
        try:
            import requests, os
            token = os.getenv("HF_TOKEN", "")
            if not token:
                from src.generator import LLMGenerator
                token = LLMGenerator()._get_hf_token()
            
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5"
            res = requests.post(url, json={"inputs": text}, headers=headers, timeout=120)
            if res.status_code == 200:
                vec = res.json()
                if isinstance(vec, list) and len(vec) == 1024:
                    arr = np.array(vec, dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    return arr.tolist()
        except Exception as exc:
            print(f"[FirestoreVectorStore] Cloud embedding notice: {exc}")
        return []

    def _call_hf_cloud_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for multiple texts in a single API call (much faster)."""
        _ensure_numpy()
        if not texts:
            return []
        try:
            import requests, os
            token = os.getenv("HF_TOKEN", "")
            if not token:
                from src.generator import LLMGenerator
                token = LLMGenerator()._get_hf_token()
            
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5"
            # Batch API: send all texts at once
            res = requests.post(url, json={"inputs": texts}, headers=headers, timeout=120)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) == len(texts):
                    results = []
                    for vec in data:
                        if isinstance(vec, list) and len(vec) == 1024:
                            arr = np.array(vec, dtype=np.float32)
                            norm = np.linalg.norm(arr)
                            if norm > 0:
                                arr = arr / norm
                            results.append(arr.tolist())
                        else:
                            results.append([0.0] * EMBEDDING_DIM)
                    return results
        except Exception as exc:
            print(f"[FirestoreVectorStore] Batch embedding notice: {exc}")
        # Fallback: one by one
        return [self._call_hf_cloud_embedding(t) for t in texts]

    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Return a list of 1024-dim float lists using HF Cloud Feature Extraction API (batch for speed)."""
        if not texts:
            return []
        
        # Use batch API for multiple texts (much faster)
        if len(texts) > 1:
            results = self._call_hf_cloud_embedding_batch(texts)
            if results and any(any(x != 0.0 for x in v) for v in results):
                return results
        
        # Fallback: one by one
        results = []
        for text in texts:
            vec = self._call_hf_cloud_embedding(text)
            if not vec:
                if CLOUD_ONLY_MODE:
                    vec = [0.0] * EMBEDDING_DIM
                else:
                    vec = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0].tolist()
            results.append(vec)
        return results

    _query_cache: Dict[str, List[float]] = {}

    def _compute_query_embedding(self, query: str) -> List[float]:
        """Encode query string directly via HF Cloud Feature Extraction API (cached)."""
        if query in self._query_cache:
            return self._query_cache[query]
        vec = self._call_hf_cloud_embedding(query)
        if not vec:
            if CLOUD_ONLY_MODE:
                print("[FirestoreVectorStore] CLOUD_ONLY_MODE — cloud query embedding failed, returning zero vector")
                vec = [0.0] * EMBEDDING_DIM
            else:
                # Fallback to local model only if cloud API unavailable
                arr = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                vec = arr.tolist()
        if vec and any(x != 0.0 for x in vec):
            self._query_cache[query] = vec
        return vec

    # ── Write ─────────────────────────────────────────────────────────────────

    def upload_policy_chunks(
        self,
        chunks: List[Any],
        force_reload: bool = False,
    ) -> None:
        """
        Batch-upsert LangChain Document chunks into Firestore and in-memory fallback.
        """
        _ensure_numpy()
        if force_reload:
            self.clear()

        if not chunks:
            print("[FirestoreVectorStore] No chunks to upload.")
            return

        # Build document list
        docs_to_upsert: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                content = str(chunk.get("chunk_text") or chunk.get("content") or chunk.get("page_content", ""))
                meta = chunk.get("metadata")
                if not isinstance(meta, dict):
                    meta = {k: v for k, v in chunk.items() if k not in ("chunk_text", "content", "page_content")}
                doc_id = str(chunk.get("doc_id") or _chunk_id(content, meta, i))
            else:
                meta = getattr(chunk, "metadata", {})
                content = getattr(chunk, "page_content", "")
                doc_id = _chunk_id(content, meta, i)

            if not content:
                continue

            docs_to_upsert.append(
                {
                    "doc_id": doc_id,
                    "title": str(meta.get("filename") or f"doc_{i}"),
                    "content": content,
                    "metadata": {
                        "company": str(meta.get("company") or meta.get("company_id") or "General"),
                        "company_id": str(meta.get("company_id") or meta.get("company") or "General"),
                        "subfolder": str(meta.get("subfolder") or "General"),
                        "subtopic": str(meta.get("subtopic") or meta.get("subfolder") or "General"),
                        "filename": str(meta.get("filename") or meta.get("document_id") or f"doc_{i}"),
                        "document_id": str(meta.get("document_id") or meta.get("filename") or f"doc_{i}"),
                        "source": str(meta.get("source") or meta.get("source_file") or ""),
                        "source_file": str(meta.get("source_file") or meta.get("source") or ""),
                        "page": _safe_int(meta.get("page") or meta.get("page_number"), 1),
                        "page_number": _safe_int(meta.get("page_number") or meta.get("page"), 1),
                        "section_header": str(meta.get("section_header") or ""),
                        "chunk_hash": doc_id,
                    },
                }
            )

        # Content fingerprints so re-uploads with changed content overwrite stale copies
        for d in docs_to_upsert:
            d["_content_fp"] = hashlib.sha256(d["content"].encode("utf-8")).hexdigest()[:16]

        # Identify docs missing from local memory store or with changed content
        missing_in_memory = [
            d for d in docs_to_upsert
            if d["doc_id"] not in self._memory_store
            or self._memory_store[d["doc_id"]].get("_content_fp") != d["_content_fp"]
        ]

        # Filter out docs already in Firestore with identical content; keep changed ones
        existing_ids = self._get_existing_ids([d["doc_id"] for d in docs_to_upsert])
        stale_ids = self._get_stale_ids([d for d in docs_to_upsert if d["doc_id"] in existing_ids])
        new_docs = [
            d for d in docs_to_upsert
            if d["doc_id"] not in existing_ids or d["doc_id"] in stale_ids
        ]

        # Embed any docs missing from memory store
        if missing_in_memory:
            texts = [d["content"] for d in missing_in_memory]
            embeddings = self._compute_embeddings(texts)
            for doc_data, emb in zip(missing_in_memory, embeddings):
                self._memory_store[doc_data["doc_id"]] = {
                    **doc_data,
                    "embedding": emb,
                    "uploaded_at": int(time.time()),
                }

        if not new_docs:
            print(f"[FirestoreVectorStore] All {len(chunks)} chunks cached & indexed.")
            return

        print(
            f"[FirestoreVectorStore] Writing {len(new_docs)} new chunks to Firestore..."
        )

        total_written = 0
        for batch_start in range(0, len(new_docs), self.batch_size):
            batch_docs = new_docs[batch_start : batch_start + self.batch_size]
            texts = [d["content"] for d in batch_docs]
            embeddings = [self._memory_store[d["doc_id"]]["embedding"] for d in batch_docs]

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
                    docs = self.collection.where(filter=FieldFilter("doc_id", "in", batch_ids)).stream()
                    for d in docs:
                        existing.add(d.id)
                except Exception as exc:
                    print(f"[FirestoreVectorStore] Warning checking existing IDs: {exc}")
        return existing

    def _get_stale_ids(self, docs: List[Dict[str, Any]]) -> set:
        """Return doc_ids whose Firestore content differs from the incoming docs."""
        stale: set = set()
        if not self.is_online or not docs:
            return stale
        for i in range(0, len(docs), 30):
            batch = docs[i : i + 30]
            try:
                fetched = {
                    d.id: (d.to_dict() or {}).get("content", "")
                    for d in self.collection.where(
                        filter=FieldFilter("doc_id", "in", [b["doc_id"] for b in batch])
                    ).stream()
                }
                for d in batch:
                    if d["doc_id"] in fetched and fetched[d["doc_id"]] != d["content"]:
                        stale.add(d["doc_id"])
            except Exception as exc:
                print(f"[FirestoreVectorStore] Warning checking stale IDs: {exc}")
        return stale

    # ── Read: Vector Search ───────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        company: Optional[str] = None,
        subfolder: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Public vector search method (alias for vector_search_firestore)."""
        effective_limit = limit if limit is not None else top_k
        return self.vector_search_firestore(query=query, limit=effective_limit, company=company, subfolder=subfolder)

    def _ensure_memory_store_populated(self) -> None:
        """If in-memory store is empty but Firestore is online, hydrate memory store from Firestore."""
        if self._memory_store:
            return
        if self.is_online:
            try:
                docs = self.collection.stream()
                missing_emb_docs = []
                for doc in docs:
                    data = doc.to_dict() or {}
                    doc_id = data.get("doc_id", doc.id)
                    emb_raw = data.get("embedding")
                    emb = []
                    if hasattr(emb_raw, "value"):
                        emb = list(emb_raw.value)
                    elif isinstance(emb_raw, (list, tuple)):
                        emb = [float(x) for x in emb_raw]

                    content = data.get("content", "")
                    if doc_id and content:
                        entry = {
                            "doc_id": doc_id,
                            "content": content,
                            "title": data.get("title", ""),
                            "metadata": data.get("metadata", {}),
                            "embedding": emb,
                        }
                        self._memory_store[doc_id] = entry
                        if not emb:
                            missing_emb_docs.append(entry)

                # Compute embeddings for any docs that lacked saved embeddings
                if missing_emb_docs:
                    texts = [d["content"] for d in missing_emb_docs]
                    embeddings = self._compute_embeddings(texts)
                    for doc_entry, vec in zip(missing_emb_docs, embeddings):
                        doc_entry["embedding"] = vec

            except Exception as exc:
                print(f"[FirestoreVectorStore] Memory store hydration notice: {exc}")

    def vector_search_firestore(
        self,
        query: str,
        limit: int = 20,
        company: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search via Firestore native vector index or in-memory BGE cosine search.
        """
        _ensure_numpy()
        query_vector = self._compute_query_embedding(query)

        if self.is_online and FirestoreVectorStore._has_native_vector_index is not False:
            try:
                col_ref = self.collection
                if company and company != "All Companies":
                    col_ref = col_ref.where(filter=FieldFilter("metadata.company", "==", company))
                    if subfolder and subfolder != "All Subfolders":
                        col_ref = col_ref.where(filter=FieldFilter("metadata.subfolder", "==", subfolder))

                vector_query = col_ref.find_nearest(
                    vector_field="embedding",
                    query_vector=Vector(query_vector),
                    distance_measure=DistanceMeasure.COSINE,
                    limit=limit * 2,
                )
                docs = vector_query.get()
                results = []
                for doc in docs:
                    data = doc.to_dict() or {}
                    meta = data.get("metadata", {})
                    doc_company = meta.get("company") or meta.get("company_id") or "General"
                    doc_subfolder = meta.get("subfolder") or "General"
                    doc_id = data.get("doc_id", getattr(doc, "id", ""))
                    filename_val = meta.get("filename") or meta.get("document_id") or "doc"
                    page_val = _safe_int(meta.get("page") or meta.get("page_number"), 1)

                    if company and company != "All Companies":
                        if doc_company != company:
                            continue
                        if subfolder and subfolder != "All Subfolders":
                            if doc_subfolder != subfolder:
                                continue

                    raw_dist = getattr(doc, "distance", None)
                    if raw_dist is None:
                        raw_dist = getattr(doc, "vector_distance", None)
                    if raw_dist is None:
                        raw_dist = data.get("distance", 0.0)

                    distance = float(raw_dist or 0.0)
                    similarity = max(0.0, 1.0 - distance)
                    results.append(
                        {
                            "doc_id": doc_id,
                            "chunk_hash": doc_id,
                            "chunk_text": data.get("content", ""),
                            "source_file": meta.get("source") or meta.get("source_file") or "—",
                            "company": doc_company,
                            "company_id": doc_company,
                            "subfolder": doc_subfolder,
                            "subtopic": meta.get("subtopic") or doc_subfolder,
                            "filename": filename_val,
                            "document_id": filename_val,
                            "page": page_val,
                            "page_number": page_val,
                            "section_header": meta.get("section_header") or "",
                            "distance": distance,
                            "score": similarity,
                            "metadata": meta,
                        }
                    )
                    if len(results) >= limit:
                        break
                if results:
                    FirestoreVectorStore._has_native_vector_index = True
                    return results
            except Exception as exc:
                FirestoreVectorStore._has_native_vector_index = False
                if not FirestoreVectorStore._printed_index_notice:
                    FirestoreVectorStore._printed_index_notice = True
                    print("[FirestoreVectorStore] Using high-performance in-memory vectorized search engine (cached & indexed from Firestore).")

        # Ensure memory store is populated from Firestore if native search failed or returned empty
        self._ensure_memory_store_populated()

        # Vectorized matrix dot-product operations on normalized embeddings
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        filtered_docs = []
        doc_embeddings = []

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
            filtered_docs.append((doc_data, doc_company, doc_subfolder, meta))
            doc_embeddings.append(emb)

        if not filtered_docs:
            return []

        # Vectorized Matrix Product Q . D^T
        emb_matrix = np.array(doc_embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized_matrix = emb_matrix / norms

        similarities = np.dot(normalized_matrix, q_vec)

        scored_docs = []
        for idx, (doc_data, doc_company, doc_subfolder, meta) in enumerate(filtered_docs):
            sim = float(similarities[idx])
            doc_id = doc_data["doc_id"]
            filename_val = meta.get("filename", meta.get("document_id", ""))
            page_val = meta.get("page_number", meta.get("page", 1))
            scored_docs.append(
                {
                    "doc_id": doc_id,
                    "chunk_hash": doc_id,
                    "chunk_text": doc_data.get("content", ""),
                    "source_file": meta.get("source", meta.get("source_file", "—")),
                    "company": doc_company,
                    "company_id": doc_company,
                    "subfolder": doc_subfolder,
                    "subtopic": meta.get("subtopic", doc_subfolder),
                    "filename": filename_val,
                    "document_id": filename_val,
                    "page": page_val,
                    "page_number": page_val,
                    "section_header": meta.get("section_header", ""),
                    "score": max(0.0, sim),
                    "metadata": meta,
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
        _ensure_numpy()
        if self.is_online:
            try:
                query = self.collection
                if company and company != "All Companies":
                    query = query.where(filter=FieldFilter("metadata.company", "==", company))
                    if subfolder and subfolder != "All Subfolders":
                        query = query.where(filter=FieldFilter("metadata.subfolder", "==", subfolder))
                docs = query.stream()
                results = []
                for doc in docs:
                    data = doc.to_dict() or {}
                    meta = data.get("metadata", {})
                    doc_id = data.get("doc_id", doc.id)
                    company_val = meta.get("company") or meta.get("company_id") or "General"
                    subfolder_val = meta.get("subfolder") or "General"
                    filename_val = meta.get("filename") or meta.get("document_id") or "doc"
                    page_val = _safe_int(meta.get("page") or meta.get("page_number"), 1)
                    content = data.get("content", "")

                    # Cache into memory store for fast local access
                    if doc_id and content:
                        emb_raw = data.get("embedding")
                        emb = list(emb_raw.value) if hasattr(emb_raw, "value") else (list(emb_raw) if isinstance(emb_raw, (list, tuple)) else [])
                        self._memory_store[doc_id] = {
                            "doc_id": doc_id,
                            "content": content,
                            "title": data.get("title", ""),
                            "metadata": meta,
                            "embedding": emb,
                        }

                    results.append(
                        {
                            "doc_id": doc_id,
                            "chunk_hash": doc_id,
                            "chunk_text": content,
                            "keywords": data.get("keywords") or _tokenize(content)[:200],
                            "source_file": meta.get("source") or meta.get("source_file") or "—",
                            "company": company_val,
                            "company_id": company_val,
                            "subfolder": subfolder_val,
                            "subtopic": meta.get("subtopic") or subfolder_val,
                            "filename": filename_val,
                            "document_id": filename_val,
                            "page": page_val,
                            "page_number": page_val,
                            "section_header": meta.get("section_header") or "",
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
            doc_company = meta.get("company") or meta.get("company_id") or "General"
            doc_subfolder = meta.get("subfolder") or "General"
            doc_id = data.get("doc_id", "")
            filename_val = meta.get("filename") or meta.get("document_id") or "doc"
            page_val = _safe_int(meta.get("page") or meta.get("page_number"), 1)

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
                    "keywords": data.get("keywords") or _tokenize(data.get("content", ""))[:200],
                    "source_file": meta.get("source") or meta.get("source_file") or "—",
                    "company": doc_company,
                    "company_id": doc_company,
                    "subfolder": doc_subfolder,
                    "subtopic": meta.get("subtopic") or doc_subfolder,
                    "filename": filename_val,
                    "document_id": filename_val,
                    "page": page_val,
                    "page_number": page_val,
                    "section_header": meta.get("section_header") or "",
                    "score": 0.0,
                    "metadata": meta,
                }
            )
        return results

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_company_chunks(self, company: str) -> None:
        """Delete all documents belonging to a company."""
        _ensure_numpy()
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
                self._batch_delete(self.collection.where(filter=FieldFilter("metadata.company", "==", company)))
            except Exception as exc:
                print(f"[FirestoreVectorStore] delete_company_chunks error: {exc}")
        print(f"[FirestoreVectorStore] Deleted all chunks for company '{company}'.")

    def delete_subfolder_chunks(self, company: str, subfolder: str) -> None:
        """Delete all documents belonging to a company/subfolder."""
        _ensure_numpy()
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
                    .where(filter=FieldFilter("metadata.company", "==", company))
                    .where(filter=FieldFilter("metadata.subfolder", "==", subfolder))
                )
                self._batch_delete(query)
            except Exception as exc:
                print(f"[FirestoreVectorStore] delete_subfolder_chunks error: {exc}")
        print(f"[FirestoreVectorStore] Deleted chunks for {company}/{subfolder}.")

    def delete_file_chunks(self, source_path: str) -> None:
        """Delete all documents belonging to a specific source file."""
        _ensure_numpy()
        from pathlib import Path as _Path
        try:
            source_str = str(_Path(source_path).resolve())
        except Exception:
            source_str = str(source_path)

        to_del = []
        for doc_id, d in self._memory_store.items():
            meta = d.get("metadata", {})
            meta_src = meta.get("source", meta.get("source_file", ""))
            try:
                meta_src_norm = str(_Path(meta_src).resolve())
            except Exception:
                meta_src_norm = str(meta_src)
            if meta_src_norm == source_str or meta_src == source_path or meta.get("filename") == _Path(source_path).name:
                to_del.append(doc_id)

        for doc_id in to_del:
            del self._memory_store[doc_id]

        if self.is_online:
            try:
                self._batch_delete(self.collection.where(filter=FieldFilter("metadata.source", "==", source_str)))
                self._batch_delete(self.collection.where(filter=FieldFilter("metadata.source_file", "==", source_str)))
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
        batch_size = self.batch_size
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
        """Return document count cross-version safely."""
        if self.is_online:
            try:
                agg = self.collection.count()
                result = agg.get()
                if result:
                    # Unwrap nested lists if present e.g. [[AggregationResult...]]
                    items = result[0] if isinstance(result, list) and len(result) > 0 else result
                    if isinstance(items, list) and len(items) > 0:
                        items = items[0]
                    if isinstance(items, (int, float)):
                        return int(items)
                    if isinstance(items, dict):
                        val = items.get("value") or items.get("count")
                        if val is not None:
                            return int(val)
                    if hasattr(items, "value") and getattr(items, "value") is not None:
                        try:
                            return int(items.value)
                        except (ValueError, TypeError):
                            pass
                    if isinstance(items, (list, tuple)) and len(items) > 0:
                        first_val = items[0]
                        v = getattr(first_val, "value", None)
                        if v is not None:
                            try:
                                return int(v)
                            except (ValueError, TypeError):
                                pass
            except Exception as exc:
                print(f"[FirestoreVectorStore] count query notice: {exc}")

        if not self._memory_store:
            self._ensure_memory_store_populated()
        return len(self._memory_store)


# ── FAISS In-Memory Vector Index ──────────────────────────────────────────────

class FAISSIndex:
    """
    Fast in-memory vector search using FAISS.
    Loads embeddings from Firestore on startup for instant queries.
    """

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self._index = None  # type: ignore
        self._doc_ids: List[str] = []
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _ensure_faiss(self):
        """Lazy-import FAISS on first use."""
        global faiss
        if faiss is None:
            try:
                import faiss as _faiss
                faiss = _faiss
            except ImportError:
                raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    def load_from_firestore(self, store: FirestoreVectorStore) -> int:
        """
        Load all embeddings from Firestore into FAISS index.
        Returns number of vectors loaded.
        """
        self._ensure_faiss()
        if not store.is_online:
            print("[FAISSIndex] Firestore offline — cannot load index")
            return 0

        try:
            # First, hydrate memory store so embeddings are available
            store._ensure_memory_store_populated()
            
            # Use _memory_store which has embeddings cached
            vectors = []
            doc_ids = []
            metadata = []

            for doc_id, data in store._memory_store.items():
                emb = data.get("embedding", [])
                if emb and len(emb) == self.dimension and doc_id:
                    vectors.append(emb)
                    doc_ids.append(doc_id)
                    metadata.append(data.get("metadata", {}))

            if not vectors:
                print("[FAISSIndex] No valid embeddings found in memory store")
                return 0

            # Build FAISS index (Inner Product for cosine similarity after normalization)
            import numpy as _np
            matrix = _np.array(vectors, dtype=_np.float32)
            # Normalize for cosine similarity
            norms = _np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = _np.where(norms == 0, 1.0, norms)
            matrix = matrix / norms

            self._index = faiss.IndexFlatIP(self.dimension)
            self._index.add(matrix)
            self._doc_ids = doc_ids
            self._metadata = {doc_id: meta for doc_id, meta in zip(doc_ids, metadata)}
            self._loaded = True

            print(f"[FAISSIndex] Loaded {len(vectors)} vectors from Firestore")
            return len(vectors)

        except Exception as exc:
            print(f"[FAISSIndex] Load error: {exc}")
            return 0

    def add_vectors(self, vectors: List[List[float]], doc_ids: List[str],
                    metadata_list: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add new vectors to the FAISS index."""
        self._ensure_faiss()
        if not vectors:
            return

        import numpy as _np
        matrix = _np.array(vectors, dtype=_np.float32)
        norms = _np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = _np.where(norms == 0, 1.0, norms)
        matrix = matrix / norms

        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dimension)

        self._index.add(matrix)
        self._doc_ids.extend(doc_ids)
        if metadata_list:
            for doc_id, meta in zip(doc_ids, metadata_list):
                self._metadata[doc_id] = meta
        self._loaded = True
        print(f"[FAISSIndex] Added {len(vectors)} vectors (total: {self._index.ntotal})")

    def search(self, query_vector: List[float], top_k: int = 10,
               company: Optional[str] = None, subfolder: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search FAISS index for nearest neighbors.
        Returns list of {doc_id, score, metadata}.
        """
        self._ensure_faiss()
        if self._index is None or self._index.ntotal == 0:
            return []

        import numpy as _np
        q = _np.array([query_vector], dtype=_np.float32)
        q_norm = _np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Search more than needed to allow for filtering
        search_k = min(top_k * 5, self._index.ntotal)
        scores, indices = self._index.search(q, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._doc_ids):
                continue
            doc_id = self._doc_ids[idx]
            meta = self._metadata.get(doc_id, {})

            # Filter by company if specified
            if company and meta.get("company", "") != company:
                continue
            
            # Filter by subfolder (policy category) if specified
            if subfolder and meta.get("subfolder", "") != subfolder:
                continue

            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "metadata": meta,
            })

            if len(results) >= top_k:
                break

        return results

    def clear(self) -> None:
        """Clear the FAISS index."""
        self._index = None
        self._doc_ids = []
        self._metadata = {}
        self._loaded = False
        print("[FAISSIndex] Index cleared")

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._index is not None and self._index.ntotal > 0

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0


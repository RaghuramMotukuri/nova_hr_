"""
vectorstore.py
ChromaDB-backed dense vector store using SentenceTransformers.
Uses content-hash IDs so upsert is idempotent and new documents
are always picked up without wiping the collection.
Supports metadata filtering by company and subfolder.
"""
import hashlib
from pathlib import Path
from typing import List, Any, Dict

import chromadb
from chromadb.api.types import Include
from sentence_transformers import SentenceTransformer


# Fields we always want back from ChromaDB queries
_INCLUDE: Include = ["documents", "distances", "metadatas"]  # type: ignore[assignment]


class ChromaVectorStore:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        db_path: str = "./chroma_db",
        collection_name: str = "lova_hr_docs",
    ):
        self.model = SentenceTransformer(model_name)
        print(f"[VectorStore] Loaded embedding model: {model_name}")

        self.collection_name = collection_name
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        print(
            f"[VectorStore] Connected to ChromaDB at '{db_path}' "
            f"collection '{collection_name}' ({self.collection.count()} items)."
        )

    @property
    def collection(self):
        """Always return active collection handle from ChromaDB (never holds stale UUIDs)."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _chunk_id(text: str, metadata: Dict[str, Any] | None = None, index: int = 0) -> str:
        """Stable, unique ID per document chunk."""
        meta = metadata or {}
        source = str(meta.get("source", ""))
        company = str(meta.get("company", ""))
        subfolder = str(meta.get("subfolder", ""))
        page = str(meta.get("page", ""))
        raw = f"{source}:{company}:{subfolder}:{page}:{index}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def clear(self) -> None:
        """Drop and recreate the collection (full re-index)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        print("[VectorStore] Collection cleared.")

    def delete_file_chunks(self, source_path: str) -> None:
        """Delete all chunks belonging to a specific source path from ChromaDB."""
        try:
            source_str = str(Path(source_path).resolve())
        except Exception:
            source_str = str(source_path)
        try:
            self.collection.delete(where={"source": source_str})  # type: ignore[arg-type]
            print(f"[VectorStore] Deleted chunks for source: {source_str}")
            print(f"[VectorStore] Current count: {self.collection.count()} items.")
        except Exception as e:
            print(f"[VectorStore] Error deleting file chunks for '{source_str}': {e}")

    def delete_company_chunks(self, company: str) -> None:
        """Delete all chunks belonging to a company from ChromaDB."""
        if not company or company == "All Companies":
            return
        try:
            self.collection.delete(where={"company": company})  # type: ignore[arg-type]
            print(f"[VectorStore] Deleted chunks for company: '{company}'")
        except Exception as e:
            print(f"[VectorStore] Error deleting company '{company}': {e}")

    def delete_subfolder_chunks(self, company: str, subfolder: str) -> None:
        """Delete all chunks belonging to a subfolder of a company from ChromaDB."""
        if not company or company == "All Companies" or not subfolder or subfolder == "All Subfolders":
            return
        try:
            self.collection.delete(
                where={
                    "$and": [
                        {"company": company},
                        {"subfolder": subfolder}
                    ]
                }
            )  # type: ignore[arg-type]
            print(f"[VectorStore] Deleted chunks for {company}/{subfolder}")
        except Exception as e:
            print(f"[VectorStore] Error deleting {company}/{subfolder}: {e}")

    # ── Write ─────────────────────────────────────────────────────────────────
    def sync_chunks(self, chunks: List[Any], force_reload: bool = False) -> None:
        """
        Upsert only NEW chunks into ChromaDB (skips already-stored ones).
        - Uses content-hash IDs -> safe to call repeatedly (idempotent).
        - force_reload=True clears the collection first.
        """
        if force_reload:
            self.clear()

        if not chunks:
            print("[VectorStore] No chunks to sync.")
            return

        texts: List[str] = [chunk.page_content for chunk in chunks]
        ids: List[str]   = [
            self._chunk_id(chunk.page_content, getattr(chunk, "metadata", {}), i)
            for i, chunk in enumerate(chunks)
        ]
        metadatas: List[Dict[str, str]] = [
            {
                "source": str(getattr(chunk, "metadata", {}).get("source", f"doc_{i}")),
                "company": str(getattr(chunk, "metadata", {}).get("company", "General")),
                "subfolder": str(getattr(chunk, "metadata", {}).get("subfolder", "General")),
                "filename": str(getattr(chunk, "metadata", {}).get("filename", ""))
            }
            for i, chunk in enumerate(chunks)
        ]

        # ── Skip chunks already in ChromaDB (avoids re-embedding on restart) ──
        try:
            unique_ids = list(set(ids))
            existing = set(self.collection.get(ids=unique_ids)["ids"]) if unique_ids else set()
        except Exception as get_exc:
            print(f"[VectorStore] Notice during get check ({get_exc}) - proceeding with fresh index")
            existing = set()
        new_indices = [i for i, cid in enumerate(ids) if cid not in existing]

        if not new_indices:
            print(f"[VectorStore] All {len(chunks)} chunks already indexed - skipping embedding.")
            return

        # Deduplicate new_indices to ensure unique IDs are passed to upsert
        seen_new_ids = set()
        unique_new_indices = []
        for idx in new_indices:
            cid = ids[idx]
            if cid not in seen_new_ids:
                seen_new_ids.add(cid)
                unique_new_indices.append(idx)

        new_texts     = [texts[i]     for i in unique_new_indices]
        new_ids       = [ids[i]       for i in unique_new_indices]
        new_metadatas = [metadatas[i] for i in unique_new_indices]

        print(f"[VectorStore] Embedding {len(new_ids)} new chunks (skipping {len(existing)} existing)...")
        embeddings = self.model.encode(
            new_texts, batch_size=64, show_progress_bar=True
        ).tolist()

        self.collection.upsert(
            ids=new_ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=new_texts,
            metadatas=new_metadatas,  # type: ignore[arg-type]
        )
        print(f"[VectorStore] Sync complete - {self.collection.count()} items stored.")

    # ── Read ──────────────────────────────────────────────────────────────────
    def query_dense(
        self,
        query: str,
        n_results: int,
        company: str | None = None,
        subfolder: str | None = None,
    ) -> Dict[str, Any]:
        """
        Query ChromaDB with a dense vector and optional metadata filters,
        returning a dict containing 'documents', 'distances', and 'metadatas'.
        """
        try:
            count = self.collection.count()
        except Exception:
            count = 0

        n_results = min(n_results, count)

        # Guard: collection is empty or n_results reduced to 0
        if n_results <= 0:
            return {"documents": [[]], "distances": [[]], "metadatas": [[]]}

        query_embedding: List[float] = self.model.encode([query])[0].tolist()

        # Build metadata filter
        where_filter: Dict[str, Any] = {}
        if company and company != "All Companies":
            if subfolder and subfolder != "All Subfolders":
                where_filter = {
                    "$and": [
                        {"company": company},
                        {"subfolder": subfolder}
                    ]
                }
            else:
                where_filter = {"company": company}

        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter if where_filter else None,  # type: ignore[arg-type]
                include=_INCLUDE,
            )
        except Exception as e:
            print(f"[VectorStore] Query warning ({e})")
            return {"documents": [[]], "distances": [[]], "metadatas": [[]]}

        # Normalise to plain dict so callers don't depend on QueryResult internals
        return {
            "documents": result.get("documents") or [[]],
            "distances":  result.get("distances") or [[]],
            "metadatas":  result.get("metadatas") or [[]],
        }
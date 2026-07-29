"""
vectorstore.py
ChromaDB-backed dense vector store using SentenceTransformers.
Uses content-hash IDs so upsert is idempotent and new documents
are always picked up without wiping the collection.
Supports metadata filtering by company and subfolder.
"""
import hashlib
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

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"[VectorStore] Connected to collection '{collection_name}' "
            f"at '{db_path}' ({self.collection.count()} items)."
        )

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _chunk_id(text: str) -> str:
        """Stable, content-derived ID so upsert is idempotent."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def clear(self) -> None:
        """Drop and recreate the collection (full re-index)."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        print("[VectorStore] Collection cleared.")

    def delete_file_chunks(self, source_path: str) -> None:
        """Delete all chunks belonging to a specific source path from ChromaDB."""
        source_str = str(source_path)
        self.collection.delete(where={"source": source_str})
        print(f"[VectorStore] Deleted chunks for source: {source_str}")
        print(f"[VectorStore] Current count: {self.collection.count()} items.")

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
        ids: List[str]   = [self._chunk_id(t) for t in texts]
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
        existing = set(self.collection.get(ids=ids)["ids"])
        new_indices = [i for i, cid in enumerate(ids) if cid not in existing]

        if not new_indices:
            print(f"[VectorStore] All {len(chunks)} chunks already indexed - skipping embedding.")
            return

        new_texts     = [texts[i]     for i in new_indices]
        new_ids       = [ids[i]       for i in new_indices]
        new_metadatas = [metadatas[i] for i in new_indices]

        print(f"[VectorStore] Embedding {len(new_indices)} new chunks (skipping {len(existing)} existing)...")
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
        count = self.collection.count()
        n_results = min(n_results, count)

        # Guard: collection is empty or n_results reduced to 0
        if n_results <= 0:
            return {"documents": [[]], "distances": [[]], "metadatas": [[]]}

        query_embedding: List[float] = self.model.encode([query])[0].tolist()

        # Build metadata filter
        where_filter = {}
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

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter if where_filter else None,
            include=_INCLUDE,
        )

        # Normalise to plain dict so callers don't depend on QueryResult internals
        return {
            "documents": result.get("documents") or [[]],
            "distances":  result.get("distances") or [[]],
            "metadatas":  result.get("metadatas") or [[]],
        }
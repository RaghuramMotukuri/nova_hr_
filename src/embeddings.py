"""
embeddings.py
EmbeddingPipeline — chunking, vector-store sync,
and scoped lexical and semantic search.
"""
from typing import List, Any, Dict
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

try:
    # Relative import when used as a package (e.g. Pylance, direct module use)
    from .vectorstore import ChromaVectorStore
except ImportError:
    # Absolute import when run from project root via Streamlit
    from src.vectorstore import ChromaVectorStore  # type: ignore[no-redef]


import re

class EmbeddingPipeline:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = ChromaVectorStore()

    # ── Chunking ──────────────────────────────────────────────────────────────
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"[Pipeline] {len(documents)} docs -> {len(chunks)} chunks")
        return chunks

    # ── Vector store ──────────────────────────────────────────────────────────
    def sync_vector_store(self, chunks: List[Any]) -> None:
        """Upsert chunks into ChromaDB (idempotent via content-hash IDs)."""
        self.vector_store.sync_chunks(chunks)

    # ── Lexical Search (BM25) ─────────────────────────────────────────────────
    def lexical_search(
        self,
        query: str,
        chunks: List[Any],
        top_k: int = 3,
        company: str | None = None,
        subfolder: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Run scoped BM25 search over a filtered subset of chunks."""
        if not chunks:
            return []

        # Filter chunks based on company and subfolder scope
        filtered_chunks = []
        for chunk in chunks:
            c = chunk.metadata.get("company", "General")
            s = chunk.metadata.get("subfolder", "General")

            if company and company != "All Companies":
                if c != company:
                    continue
                if subfolder and subfolder != "All Subfolders":
                    if s != subfolder:
                        continue

            filtered_chunks.append(chunk)

        if not filtered_chunks:
            return []

        # Tokenize using regex to clean punctuation
        def tokenize(text: str) -> List[str]:
            return re.findall(r"\w+", text.lower())

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Build dynamic BM25 index on filtered chunks
        corpus = [tokenize(chunk.page_content) for chunk in filtered_chunks]
        bm25_index = BM25Okapi(corpus)

        # Clip negative IDF values to a small positive epsilon (0.0001) to prevent negative BM25 scores
        # and reverse ranking bugs in small corpora.
        if hasattr(bm25_index, "idf"):
            bm25_index.idf = {k: max(v, 0.0001) for k, v in bm25_index.idf.items()}

        scores = bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            # Only include chunks with positive relevance score
            if score <= 0.0:
                continue
            chunk = filtered_chunks[idx]
            source = str(getattr(chunk, "metadata", {}).get("source", "—"))
            results.append(
                {
                    "chunk_text": chunk.page_content,
                    "source_file": source,
                    "company": chunk.metadata.get("company", "General"),
                    "subfolder": chunk.metadata.get("subfolder", "General"),
                    "filename": chunk.metadata.get("filename", ""),
                    "score": score,
                }
            )
        return results

    # ── Semantic Search (Dense Vector) ────────────────────────────────────────
    def semantic_search(
        self,
        query: str,
        top_k: int = 3,
        company: str | None = None,
        subfolder: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB with optional metadata scoping and return top-k dense matches."""
        try:
            chroma_res = self.vector_store.query_dense(
                query,
                n_results=top_k,
                company=company,
                subfolder=subfolder
            )
        except Exception as e:
            print(f"[Pipeline] VectorStore error ({e}) - refreshing store instance...")
            try:
                self.vector_store = ChromaVectorStore()
                chroma_res = self.vector_store.query_dense(
                    query,
                    n_results=top_k,
                    company=company,
                    subfolder=subfolder
                )
            except Exception as sub_e:
                print(f"[Pipeline] VectorStore retry failed: {sub_e}")
                chroma_res = {"documents": [[]], "distances": [[]], "metadatas": [[]]}
        
        documents = chroma_res.get("documents", [[]])[0]
        distances = chroma_res.get("distances", [[]])[0]
        metadatas = chroma_res.get("metadatas", [[]])[0]

        results = []
        for doc, dist, meta in zip(documents, distances, metadatas):
            similarity = 1.0 - float(dist)
            source = meta.get("source", "—")
            results.append(
                {
                    "chunk_text": doc,
                    "source_file": source,
                    "company": meta.get("company", "General"),
                    "subfolder": meta.get("subfolder", "General"),
                    "filename": meta.get("filename", ""),
                    "score": similarity,
                }
            )
        return results
# LOVA_HR ⚡

**Firebase-Powered Hybrid RAG Engine — BM25 + Firestore Vector Search + BGE Reranker**

LOVA_HR is a high-performance, hallucination-free document RAG application built on **Google Cloud Firestore** as the online vector database, **BAAI/bge-m3** for 1024-dimensional dense embeddings, **BM25 Okapi** for lexical retrieval, **Reciprocal Rank Fusion (RRF)** for hybrid result merging, and **BAAI/bge-reranker-v2-m3** for cross-encoder reranking.

It uses **zero generative LLMs** — every answer is grounded in verbatim retrieved document chunks, guaranteeing 100% factual accuracy with zero hallucinations.

---

## Key Features

- 🔥 **Google Cloud Firestore** — Online, scalable vector store with native 1024-dim cosine similarity search.
- 🧠 **BAAI/bge-m3 Embeddings** — State-of-the-art 1024-dimensional multilingual dense vectors.
- 🔤 **BM25 Lexical Search** — Exact keyword matching for policy section numbers, form IDs, and legal jargon.
- ⚡ **RRF Hybrid Fusion** — Reciprocal Rank Fusion merges semantic + lexical results for best-of-both retrieval.
- 🏆 **BGE Reranker V2 M3** — Cross-encoder reranking scores the top-15 fused candidates to surface the top-5 most relevant chunks.
- 📁 **Dynamic Document Library** — Upload `.txt`, `.pdf`, `.docx` files through the web UI.
- ⚖️ **Three-Tab Results View** — BM25 | Semantic (BGE-M3) | Reranked (RRF + BGE) for full pipeline transparency.
- 🔒 **Company + Subfolder Scoping** — Searches are always scoped to the selected company and subfolder.
- 💻 **Offline Fallback** — Gracefully degrades to BM25-only mode if Firebase is unreachable.

---

## Architecture

```
[ User Documents (.txt / .pdf / .docx) ]
                    │
                    ▼
           [ Text Chunking (800 chars / 150 overlap) ]
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   [ BGE-M3 Embeddings ]   [ Keywords ]
         │                     │
         ▼                     │
  [ Firestore Vector Store ]   │
  (1024-dim cosine index)      │
         │                     │
         └──────────┬──────────┘
                    │
            [ User Query ]
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
  [ Firestore find_nearest ]  [ BM25 Okapi ]
    Branch A: Semantic          Branch B: Lexical
    (Top-20)                    (Top-20)
         │                     │
         └──────────┬──────────┘
                    ▼
           [ RRF Fusion: Σ 1/(60 + rank) ]
                    │
             [ Top-15 candidates ]
                    │
        [ BGE Reranker V2 M3 CrossEncoder ]
                    │
             [ Top-5 Reranked Chunks ]
                    │
          [ Streamlit 3-Tab UI ]
     BM25 | Semantic | 🏆 Reranked
```

---

## Setup

### 1. Set Up Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First-time install downloads `BAAI/bge-m3` (~570 MB) and `BAAI/bge-reranker-v2-m3` (~580 MB) from HuggingFace.

### 3. Configure Firebase Credentials

**Option A — Local file (recommended for development):**
1. Go to [Firebase Console](https://console.firebase.google.com) → Project Settings → Service Accounts
2. Click **Generate new private key** → save as `serviceAccountKey.json` in the repo root.

**Option B — Environment variable (recommended for CI/CD):**
```bash
export FIREBASE_SERVICE_ACCOUNT_KEY='<paste contents of serviceAccountKey.json>'
```

Copy the template:
```bash
cp .env.example .env   # then fill in your values
```

### 4. Create Firestore Vector Index

Run this **once** before first use (index creation takes 5–15 minutes):

```bash
gcloud firestore indexes composite create \
  --project=YOUR_PROJECT_ID \
  --collection-group=hr_policies \
  --query-scope=COLLECTION \
  --field-config field-path=embedding,vector-config='{"dimension":"1024","flat":{}}'
```

Check index status:
```bash
gcloud firestore indexes composite list --project=YOUR_PROJECT_ID
```

### 5. Ingest Documents into Firestore

```bash
# Ingest all documents from data/ directory
python ingest_firebase.py

# Clear existing data and re-ingest
python ingest_firebase.py --clear

# Test chunking without uploading (dry run)
python ingest_firebase.py --dry-run
```

### 6. Run the App

```bash
streamlit run app.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend UI** | Streamlit |
| **Vector Database** | Google Cloud Firestore (native vector search) |
| **Embedding Model** | `BAAI/bge-m3` (1024-dim, via FlagEmbedding) |
| **Lexical Engine** | Rank-BM25 (`BM25Okapi`) |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF, k=60) |
| **Reranker** | `BAAI/bge-reranker-v2-m3` (CrossEncoder) |
| **Document Loading** | LangChain Loaders + PyMuPDF |
| **Firebase SDK** | `firebase-admin` + `google-cloud-firestore` |

---

## Project Structure

```
lova-hr/
├── app.py                  # Streamlit UI (3-tab results: BM25 | Semantic | Reranked)
├── ingest_firebase.py      # CLI: parse docs → embed → upload to Firestore
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── serviceAccountKey.json  # 🔒 Firebase credentials (git-ignored!)
└── src/
    ├── config.py           # Firebase Admin SDK initialization
    ├── firebase_client.py  # FirestoreVectorStore (BGE-M3 + Firestore native search)
    ├── embeddings.py       # EmbeddingPipeline (BM25 + Semantic + RRF + Rerank)
    ├── retriever.py        # HybridRetriever (clean search orchestration)
    ├── data_loader.py      # PDF / TXT / DOCX loader with metadata
    └── vectorstore.py      # Legacy ChromaDB store (retained for reference)
```

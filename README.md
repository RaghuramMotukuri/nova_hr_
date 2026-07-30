<div align="center">

# ⚡ NOVA_HR

### Hybrid Retrieval-Augmented HR Policy Assistant

**Ask your HR policy documents anything. Get grounded, source-backed answers powered by state-of-the-art hybrid search and a lightweight local LLM.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Firebase](https://img.shields.io/badge/Firebase-Firestore%20Vector%20Search-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://firebase.google.com/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-BGE--M3%20%7C%20Phi--3.5-FFD21E?style=for-the-badge)](https://huggingface.co/)
[![RRF](https://img.shields.io/badge/Fusion-RRF-2E8B57?style=for-the-badge)]()

</div>

---

## 🎯 What Is This?

**NOVA_HR** is a document-grounded HR policy assistant built as a hybrid Retrieval-Augmented Generation (RAG) pipeline. Instead of leaning on a single retrieval method or a heavyweight generative model, it combines:

- **Dense semantic retrieval** with a state-of-the-art multilingual embedding model (**BGE-M3**)
- **Reciprocal Rank Fusion (RRF)** to merge multiple retrieval signals into one ranked list
- **Cross-encoder reranking** with a **BGE reranker** to sharpen precision on the top candidates
- **A small, efficient local LLM** (**Phi-3.5-mini-instruct**) to generate the final grounded answer

The result is an assistant that answers from your actual HR policy documents — with every answer traceable back to source — while staying lightweight enough to run without a large hosted model.

> 💡 **Why hybrid + rerank + fusion?** Dense retrieval alone can miss exact-term matches; fusing multiple retrieval signals via RRF captures more of the relevant candidates. A reranker then re-scores the fused list with a much stronger (but slower) cross-encoder, so only the final answer generation touches an LLM — keeping hallucination risk low and answers tightly grounded.

---

## ✨ Key Features

| | Feature | Description |
|---|---|---|
| 🔥 | **Firebase Firestore Vector Search** | Native vector search (`find_nearest`, cosine distance) — no separate vector DB infra to manage |
| 🧭 | **BGE-M3 Embeddings** | Multilingual, multi-granularity dense embeddings from Hugging Face for high-quality semantic retrieval |
| 🔀 | **RRF Fusion** | Reciprocal Rank Fusion merges multiple retrieval result sets into a single, more robust ranking |
| 🎯 | **BGE Reranker** | Cross-encoder reranking of fused candidates for precision at the top of the list |
| 🤖 | **Phi-3.5-mini-instruct** | Compact, efficient instruction-tuned LLM (via Hugging Face) generates the final grounded answer |
| 📁 | **Dynamic Document Library** | Upload and manage HR policy documents directly through the Streamlit UI |

---

## 🏗️ How It Works

```mermaid
flowchart TD
    A["📄 HR Policy Documents"] --> B["✂️ Chunking"]
    B --> C["🧭 BGE-M3 Embeddings"]
    C --> D["🔥 Firebase Firestore<br/>Vector Search"]

    Q["❓ User Query"] --> C2["🧭 Query Embedding<br/>(BGE-M3)"]
    C2 --> D

    D --> R["🔀 RRF Fusion<br/>(merge retrieval signals)"]
    R --> RR["🎯 BGE Reranker<br/>(cross-encoder re-score)"]
    RR --> L["🤖 Phi-3.5-mini-instruct<br/>(grounded answer generation)"]
    L --> UI["🖥️ Streamlit UI<br/>Answer + Sources"]

    style A fill:#1f2937,stroke:#6366f1,color:#fff
    style Q fill:#1f2937,stroke:#f59e0b,color:#fff
    style D fill:#b45309,stroke:#facc15,color:#fff
    style R fill:#0f766e,stroke:#14b8a6,color:#fff
    style RR fill:#4c1d95,stroke:#8b5cf6,color:#fff
    style L fill:#7f1d1d,stroke:#f87171,color:#fff
    style UI fill:#0369a1,stroke:#38bdf8,color:#fff
```

**The pipeline, step by step:**
1. **Ingest** — HR policy documents are chunked and embedded with **BGE-M3**.
2. **Store** — embeddings are written to **Firebase Firestore**, using its native vector search for similarity queries.
3. **Retrieve** — a user query is embedded and matched against Firestore's vector index; multiple retrieval signals are combined via **RRF fusion**.
4. **Rerank** — the fused candidate list is re-scored by a **BGE reranker** cross-encoder to surface the most relevant chunks.
5. **Generate** — the top reranked chunks are passed as context to **Phi-3.5-mini-instruct**, which generates the final answer grounded in that context.
6. **Display** — Streamlit shows the answer alongside the source chunks it was grounded in.

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend / UI** | Streamlit | Document upload, query input, answer + source display |
| **Vector Store** | Firebase Firestore | Native vector search (`find_nearest`, cosine distance) |
| **Embeddings** | BGE-M3 (Hugging Face) | Multilingual dense embeddings for semantic retrieval |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Merges multiple retrieval rankings into one |
| **Reranking** | BGE Reranker (Hugging Face) | Cross-encoder re-scoring of fused candidates |
| **Generation** | Phi-3.5-mini-instruct (Hugging Face) | Compact instruction-tuned LLM for grounded answer generation |
| **Document Loading** | LangChain Community Loaders | PDF / text ingestion pipeline |

---

## 📂 Project Structure

```
nova_hr_/
├── src/                  # Core retrieval, fusion, reranking, and generation logic
├── data/                 # Source HR policy documents
├── tests/                # Automated test suite
├── app.py                # Streamlit application entry point
├── main.py                # CLI / pipeline entry point
├── requirements.txt      # Python dependencies
└── pyproject.toml        # Project metadata
```

---

## 🚀 Getting Started

### 1️⃣ Set Up a Virtual Environment

```bash
# Create
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — macOS/Linux
source .venv/bin/activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure Firebase

Add your Firebase project credentials (service account key / config) so Firestore vector search can authenticate. See `src/` for where credentials are expected to be loaded.

### 4️⃣ Run the App

```bash
streamlit run app.py
```

Streamlit launches locally — usually at **`http://localhost:8501`**.

---

## ✅ Running Tests

```bash
pytest tests/ -v
```

---

## 🗺️ Roadmap Ideas

- [ ] Configurable RRF weighting between retrieval signals
- [ ] Swap-in support for larger Phi or Llama variants
- [ ] Multi-user document namespaces in Firestore
- [ ] Export grounded answers with citations to PDF

---

<div align="center">

**Hybrid retrieval. Grounded generation. Built for accuracy.** 🎯

</div>

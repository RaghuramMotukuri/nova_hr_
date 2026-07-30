<div align="center">

# ⚡ NOVA_HR

### Pure Lexical + Semantic Retrieval RAG Engine — **Zero LLM**

**Ask questions about your HR policy documents and get exact, verifiable answers — no hallucinations, no API keys, no cloud calls.**

![Zero LLM](https://img.shields.io/badge/LLM-ZERO-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-4A154B?style=for-the-badge)
![License](https://img.shields.io/badge/Privacy-100%25%20Local-2ea44f?style=for-the-badge)

</div>

---

## 🧭 What Is This?

Most HR chatbots wrap an LLM around your documents and hope it doesn't make something up.

**NOVA_HR does the opposite.** It never generates a single word of new text. Instead, it retrieves the *exact* chunk of your uploaded HR policy that answers the question — combining classic keyword search (BM25) with modern semantic embedding search (SentenceTransformers + ChromaDB) — and shows you both, side by side, with real scores. If it's not in the document, it doesn't invent an answer.

| Traditional LLM Chatbot | NOVA_HR |
|---|---|
| 🌀 Can hallucinate facts | ✅ 100% grounded in your documents |
| ☁️ Needs an API key + internet | 🖥️ Fully local, zero external calls |
| 💸 Per-token API cost | 🆓 Free to run, forever |
| 🔒 Sends data to a third party | 🔐 Your HR data never leaves your machine |
| 🤷 Hard to explain *why* it answered | 📊 Shows exact source chunk + score |

---

## 🏗️ The Pipeline — How a Question Becomes an Answer

```mermaid
flowchart TD
    A["📄 User Documents<br/>.txt / .pdf / .docx"] --> B["✂️ Text Chunking"]

    B --> C["🔤 BM25 Okapi Index<br/><i>Lexical / Keyword Search</i>"]
    B --> D["🧠 SentenceTransformer Embeddings<br/><i>all-MiniLM-L6-v2</i>"]
    D --> E["🗂️ ChromaDB Vector Store<br/><i>Persistent, local, incremental</i>"]

    F["❓ User Query"] --> C
    F --> D

    C --> G["🏆 Top-K Lexical Matches"]
    E --> H["🏆 Top-K Semantic Matches"]

    G --> I["⚖️ Side-by-Side Comparison View"]
    H --> I

    I --> J["🖥️ Streamlit UI<br/>Files • Chunk Text • Scores"]

    style A fill:#fff3cd,stroke:#856404
    style F fill:#d1ecf1,stroke:#0c5460
    style J fill:#d4edda,stroke:#155724
    style I fill:#f8d7da,stroke:#721c24
```

**In plain words:**

1. 📁 **You upload** HR policy documents (`.txt`, `.pdf`, `.docx`) through the web UI.
2. ✂️ Each document is **split into chunks** small enough to search precisely, large enough to keep context.
3. Every chunk is indexed **two different ways at once**:
   - 🔤 **Lexical** — BM25 builds a keyword-frequency index (great for exact terms: "12 days", "probation period").
   - 🧠 **Semantic** — MiniLM turns each chunk into a vector embedding stored in ChromaDB (great for meaning: "how much time off do I get" ≈ "leave entitlement").
4. ❓ When you ask a question, it's run through **both engines simultaneously**.
5. ⚖️ The **top matches from each side** are displayed in separate tabs — you see exactly which chunk, from which file, with its score, for both approaches.
6. 🚫 No generation step. No LLM. What you see *is* what's in the document.

---

## 🔄 Incremental Indexing — Only New Docs Get Processed

```mermaid
flowchart LR
    A["📤 New file uploaded"] --> B{"Already embedded<br/>in ChromaDB?"}
    B -- "Yes ✅" --> C["⏭️ Skip re-embedding"]
    B -- "No 🆕" --> D["✂️ Chunk → 🧠 Embed → 💾 Store"]
    E["🗑️ Delete file"] --> F["🧹 Prune from index"]

    style B fill:#fff3cd,stroke:#856404
    style D fill:#d4edda,stroke:#155724
    style F fill:#f8d7da,stroke:#721c24
```

This keeps the app fast as your document library grows — you're never re-embedding the whole knowledge base just to add one new policy PDF.

---

## 🧰 Tech Stack

```mermaid
flowchart TB
    subgraph UI["🖥️ Interface"]
        S["Streamlit"]
    end
    subgraph Ingest["📥 Document Loading"]
        L1["PyPDFLoader"]
        L2["TextLoader"]
        L3["Docx2txtLoader"]
    end
    subgraph Retrieval["🔍 Dual Retrieval Engine"]
        R1["Rank-BM25<br/>(BM25Okapi)"]
        R2["SentenceTransformers<br/>all-MiniLM-L6-v2"]
    end
    subgraph Store["💾 Storage"]
        V["ChromaDB<br/>(persistent, local: ./chroma_db)"]
    end

    S --> Ingest
    Ingest --> Retrieval
    R2 --> V
    Retrieval --> S
    V --> S
```

| Layer | Technology | Purpose |
|---|---|---|
| 🖥️ **Frontend / UI** | Streamlit | Upload docs, ask questions, view results |
| 📥 **Document Loading** | LangChain Community Loaders (`PyPDFLoader`, `TextLoader`, `Docx2txtLoader`) | Parse `.pdf` / `.txt` / `.docx` into text |
| 🧠 **Embeddings** | SentenceTransformers — `all-MiniLM-L6-v2` | Runs **locally**, converts text chunks to dense vectors |
| 🗂️ **Vector Database** | ChromaDB | Persistent local semantic index (`./chroma_db`) |
| 🔤 **Lexical Engine** | Rank-BM25 (`BM25Okapi`) | Classic keyword/term-frequency search |
| 🧪 **Testing** | Pytest | Verifies loading, indexing, search, and deletion |

---

## ✨ Feature Highlights

<table>
<tr>
<td width="50%" valign="top">

### 🧠 Zero LLM Dependency
No OpenAI, no Anthropic, no cloud generative model anywhere in the loop. No API keys, no per-query cost, no risk of the model "creatively" answering a compliance question wrong.

### 📁 Dynamic Document Library
Add or remove `.txt`, `.pdf`, `.docx` files straight from the web UI — the knowledge base updates live, no restart required.

### 🔤 Lexical Search (BM25)
Fast, exact keyword and term matching — ideal for policy numbers, exact phrases, and legal-style precision.

</td>
<td width="50%" valign="top">

### 🧠 Semantic Search (Dense Vectors)
Cosine-similarity search over embeddings — finds the right policy even when the question is phrased completely differently from the source text.

### ⚖️ Side-by-Side Comparison View
Every query shows **both** retrieval methods in responsive tabs — file name, chunk text, and precise similarity/score — so you can see *why* an answer was surfaced.

### ⚡ Incremental Vector Indexing
Already-embedded documents are automatically skipped on re-index, keeping the app fast as the library grows.

</td>
</tr>
</table>

---

## 🚀 Getting Started

```bash
# 1️⃣ Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Launch the app
streamlit run app.py
```

App opens at **`http://localhost:8501`** 🎉

### 🧪 Run the Test Suite

```bash
pytest tests/ -v
```

Covers file loading, indexing, lexical + semantic search, and incremental deletion.

---

## 📂 Project Structure

```
nova_hr_/
├── app.py              # 🖥️ Streamlit UI entry point
├── main.py             # ⚙️ Core pipeline logic
├── src/                # 🧩 Retrieval + indexing modules
├── data/                # 📄 Uploaded HR documents
├── chroma_db/           # 🗂️ Persistent vector store
├── chroma_test/         # 🧪 Test vector store
├── tests/               # ✅ Pytest suite
├── requirements.txt      # 📦 Dependencies
└── pyproject.toml        # 🔧 Project config
```

---

<div align="center">

### 🔒 Built for accuracy over eloquence.
**If it's not in the document, NOVA_HR won't tell you it is.**

</div>

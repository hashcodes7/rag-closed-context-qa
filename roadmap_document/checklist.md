# SourceIQ: Core Build Progress Tracker

## ✅ Phase 1: The Core Engine (Completed)

### Step 1 — Local Model Works (v1)
* [x] Load tokenizer and Qwen model
* [x] Basic terminal input/output loop

### Step 2 — TXT Knowledge Source (v2)
* [x] Read text from external `.txt` files

### Step 3 — Basic RAG Logic (v3)
* [x] Basic keyword-based chunk retrieval

### Step 4 — Chatbot Mode (v4)
* [x] Continuous chat loop and `quit` condition

### Step 5 — Multi-Document Citations (v5)
* [x] File-level source tracking and citations

### Step 6 — Advanced Structure & Telemetry (v6)
* [x] Sliding window chunking and metadata
* [x] Time-based telemetry logging

### Step 7 — Semantic Search & Embeddings (v7)
* [x] Vector embeddings via `sentence-transformers`
* [x] Cosine similarity retrieval

### Step 8 — Conversational Memory (v8 / v8.1)
* [x] Rolling chat history buffer
* [x] XML tagging patch for 0.5B model stability

### Step 9 — Two-Stage Retrieval (v9 / v9.1)
* [x] Cross-Encoder Reranking logic
* [x] **ChatML Patch**: Native `apply_chat_template` implementation

### Step 10 — Streaming Output (v10)
* [x] Multithreaded `TextIteratorStreamer` for real-time output

### Step 11 — Persistent Vector Caching (v11)
* [x] `torch.save/load` for instant startup from `vector_cache.pt`

---

## ✅ Phase 2: Performance & Interface (Completed)

### Step 12 — Advanced Chunking (v12)
* [x] Implement `RecursiveCharacterTextSplitter` logic
* [x] Ensure sentence-aware boundaries to prevent fact-clipping

### Step 13 — Vector Indexing (v13)
* [x] Integrate **FAISS** for indexed search
* [x] Move from linear scan to HNSW index

### Step 14 — Model Quantization (v14)
* [x] Implement 4-bit/8-bit loading via `bitsandbytes`

### Step 15 — Hybrid Search (v15)
* [x] Combine Semantic Search with Keyword Search (BM25)
* [x] Implement a reciprocal rank fusion (RRF) for scoring

### Step 16 — Streamlit Web UI (v16)
* [x] Create basic `streamlit` chat dashboard
* [x] Move terminal telemetry to visual charts

### Step 17 — Interactive UI Features (v17)
* [x] Add document upload management via sidebar
* [x] Implement "Click-to-Source" viewing in the browser

### Step 18 — Advanced File Support (v18)
* [x] Integrate `PyMuPDF` for `.pdf` ingestion
* [x] Integrate `python-docx` for `.docx` ingestion

### Step 19 — GGUF & CPU Optimization (v19)
* [x] Integrate `llama-cpp-python` for GGUF model support
* [x] Enable high-performance local inference on standard CPUs

---

## 🚀 Phase 3: The Modular RAG Lab (v20 - Current)

### Step 20 — Semantic Intelligence
* [x] **Semantic Chunking**: Context-aware splitting via sentence embeddings
* [x] **HyDE Expansion**: Hypothetical Document Embeddings for query expansion
* [x] **Parent-Document Retrieval**: Child-match with Parent-context expansion

### Step 21 — Infrastructure & Persistence
* [x] **SQLite Chat Persistence**: Conversations saved to `rag_history.db`
* [x] **Modular Algorithm Toggles**: Instant on/off controls for every pipeline stage
* [x] **Verified Citation UI**: In-text `[1], [2]` mapping to verifiable source snippets

---

# 📝 Handover Notes for Future Agents

### Current System State:
*   **Version**: `V20`
*   **Engine**: Hybrid (FAISS HNSW + BM25) with RRF blending.
*   **Intelligence**: HyDE (Optional), Semantic Chunking (Optional), Cross-Encoder Reranking (Optional).
*   **Architecture**: Modular "Toggle-Based" Lab.
*   **Persistence**: SQLite-backed history and telemetry.

### Key Files:
*   `core.py`: The RAGEngine logic.
*   `ui.py`: Streamlit interface.
*   `database.py`: SQLite persistence layer.
*   `documents/algorithms.md`: Technical documentation of all V20 algorithms.

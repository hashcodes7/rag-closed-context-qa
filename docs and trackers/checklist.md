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

# 🌌 Phase 2: Future Horizon (Performance & UI)

## Step 12 — Advanced Chunking (Performance)
* [x] Implement `RecursiveCharacterTextSplitter` logic
* [x] Ensure sentence-aware boundaries to prevent fact-clipping

## Step 13 — Vector Indexing (Performance)
* [x] Integrate **FAISS** for indexed search
* [x] Move from linear scan to HNSW (Hierarchical Navigable Small World) index

## Step 14 — Model Quantization (Performance)
* [x] Implement 4-bit/8-bit loading via `bitsandbytes`
* [x] Optimize VRAM/RAM usage for faster generation

## Step 15 — Hybrid Search (Performance)
* [ ] Combine Semantic Search with Keyword Search (BM25)
* [ ] Implement a reciprocal rank fusion (RRF) for scoring

## Step 16 — Streamlit Web UI (Interface)
* [ ] Create basic `streamlit` chat dashboard
* [ ] Move terminal telemetry to visual charts

## Step 17 — Interactive UI Features (Interface)
* [ ] Add document upload management via sidebar
* [ ] Implement "Click-to-Source" viewing in the browser

## Step 18 — Advanced File Support (Expansion)
* [ ] Integrate `PyMuPDF` for `.pdf` ingestion
* [ ] Integrate `python-docx` for `.docx` ingestion

---

# 📝 Handover Notes for Future Agents

### Current System State:
*   **Version**: `V13`
*   **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
*   **Prompting**: Must use `tokenizer.apply_chat_template` (ChatML) to prevent the 0.5B model from hallucinating.
*   **Chunking**: `recursive_chunk_text()` — separator hierarchy `\n\n` → `\n` → `". "` → `" "`, `chunk_size=1000 chars`, `overlap=200 chars`.
*   **Retrieval**: Two-stage — Stage 1: `faiss.IndexHNSWFlat` (M=32, efSearch=64) ANN search; Stage 2: Cross-Encoder `ms-marco-MiniLM-L-6-v2` reranking.
*   **Streaming**: Handled via `threading.Thread` and `TextIteratorStreamer`.
*   **Caching**: **Dual-file** — `vector_cache.pt` (chunk metadata) + `faiss_index.bin` (HNSW index). Both must exist for a cache hit. Delete both to force re-index.

### Key Files:
*   `app.py`: The main entry point and engine.
*   `knowledge_source/`: Ingest directory for data.
*   `docs and trackers/`: Historical architecture logs for every version.

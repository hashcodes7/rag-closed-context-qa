# Core Build Steps Progress Tracker

## Step 1 — Local Model Works v1

* [x] Create `app.py`
* [x] Load tokenizer
* [x] Load Qwen model
* [x] Ask one hardcoded question
* [x] Print answer in terminal

## Step 2 — TXT Knowledge Source v2

* [x] Create `notes.txt`
* [x] Read txt file in Python
* [x] Inject file text into prompt
* [x] Ask question in terminal
* [x] Answer using txt content

## Step 3 — Real RAG Logic v3

* [x] Split txt into chunks
* [x] Find relevant chunk
* [x] Pass only relevant chunk to model
* [x] Reduce hallucination

## Step 4 — Chatbot Mode v4

* [x] Infinite question loop
* [x] Ask multiple questions
* [x] Exit with `quit`

## Step 5 — Better Version v5

* [x] Multiple txt files
* [x] Save chat history
* [x] Better prompts
* [x] Source chunk display

# 🧠 RAG System Upgrade Plan (Terminal-Only)
## ✅ Upgrade Checklist

### 🟢 Level 1 — Structure Improvements (No embeddings) v6
- [x] Implement token-aware chunking (200–400 words)
- [x] Add overlap between chunks (50–100 words)
- [x] Store chunk metadata (source, chunk_id)
- [x] Replace single best match with top-k (3–5 chunks)
- [x] Improve prompt with strict grounding instructions

---

### 🟠 Level 2 — Semantic Retrieval (Embeddings) v7
- [ ] Add sentence-transformer embedding model
- [ ] Generate embeddings for all chunks at startup
- [ ] Embed user query
- [ ] Replace keyword search with cosine similarity
- [ ] Retrieve top-k semantically similar chunks

---

### 🟣 Level 3 — Retrieval Engine Upgrade (FAISS)  v8
- [ ] Integrate FAISS vector index
- [ ] Store embeddings in FAISS instead of Python list scan
- [ ] Implement fast nearest-neighbor search
- [ ] Add hybrid scoring (semantic + keyword optional)
- [ ] Add MMR to reduce redundant chunks

---

### 🔵 Level 4 — Context Intelligence Layer v9 
- [ ] Add similarity threshold filtering
- [ ] Implement query rewriting before embedding
- [ ] Improve context formatting with structured blocks
- [ ] Add optional answer verification step

---

### 🟤 Level 5 — Production-Grade CLI RAG v10
- [ ] Persist FAISS index to disk
- [ ] Load index on startup
- [ ] Add logging (query, chunks, scores, answer)
- [ ] Add debug mode for retrieval inspection
- [ ] Move hyperparameters to config file

---

# 🚀 Full Upgrade Roadmap

---

## 🟢 Level 0 — Baseline (Current System)

### Description
Current system uses:
- Line-based chunking
- Keyword overlap scoring
- Single best chunk selection
- Direct prompt injection into LLM

### Limitations
- No semantic understanding
- Weak retrieval accuracy
- Fragile keyword matching

---

## 🟢 Level 1 — Clean Chunking + Prompting

### Goal
Improve structure without changing architecture.

### Improvements

#### 1. Better chunking strategy
- Replace line splitting with token/word-based chunking
- Chunk size: 200–400 words
- Overlap: 50–100 words

#### 2. Metadata enrichment
Each chunk should store:
- source file
- chunk id
- text

#### 3. Top-k retrieval
- Instead of 1 chunk, retrieve top 3–5 chunks

#### 4. Improved prompt design
Add strict grounding rules:
- Use only context
- Say "I don't know" if missing

---

## 🟠 Level 2 — Semantic Retrieval (Embeddings)

### Goal
Replace keyword matching with meaning-based retrieval.

### Improvements

#### 1. Embedding model
- Use sentence-transformers (e.g. MiniLM)

#### 2. Chunk embeddings
- Precompute embeddings at startup

#### 3. Query embedding
- Convert user question into vector

#### 4. Similarity search
- Use cosine similarity
- Retrieve top-k chunks

### Result
- Synonym understanding
- Much better retrieval accuracy

---

## 🟣 Level 3 — Retrieval Engine (FAISS)

### Goal
Make retrieval fast and scalable.

### Improvements

#### 1. FAISS integration
- Store embeddings in FAISS index

#### 2. Fast search
- Replace loop-based search

#### 3. Hybrid scoring (optional)
- Combine semantic + keyword scores

#### 4. Diversity control (MMR)
- Avoid redundant chunks

### Result
- Production-level retrieval performance

---

## 🔵 Level 4 — Context Intelligence Layer

### Goal
Improve reasoning quality of RAG pipeline.

### Improvements

#### 1. Context filtering
- Remove low-score chunks

#### 2. Query rewriting
- Improve query before embedding

#### 3. Structured context formatting
- Add source labels and grouping

#### 4. Answer verification (optional)
- Validate response against context

### Result
- Fewer hallucinations
- More grounded answers

---

## 🟤 Level 5 — Production CLI RAG System

### Goal
Make system robust and maintainable.

### Improvements

#### 1. Persistent vector store
- Save FAISS index to disk

#### 2. Logging system
Track:
- query
- retrieved chunks
- similarity scores
- final answer

#### 3. Debug mode
- Show retrieval pipeline output

#### 4. Config-based system
- Chunk size
- top-k
- thresholds

### Result
- Fully structured CLI RAG system
- Easy debugging and tuning

---

# 🧠 Final Insight

The biggest improvement comes from:

> 🔥 Replacing keyword matching with embedding-based semantic retrieval

Everything else is incremental refinement on top of that core upgrade.

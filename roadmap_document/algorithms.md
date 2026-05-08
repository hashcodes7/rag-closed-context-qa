# 🧠 SourceIQ RAG Engine: V20 Algorithmic Blueprint

This document provides a deep dive into the mathematical and logical foundations of the SourceIQ RAG pipeline (Version 20). This version introduces modular toggles, semantic-aware indexing, and persistent intelligence.

---

## 1. Semantic Chunking (V20 Feature)
**Algorithm:** Embedding-Based Breakpoint Detection

### 📖 Description
Unlike fixed-size splitting, this algorithm analyzes the "flow" of ideas. It calculates the semantic similarity between every adjacent sentence. When the similarity drops below a threshold (e.g., 0.5), it marks a "breakpoint" and starts a new chunk.

### 🚀 Importance in RAG
Ensures that the LLM receives complete topics. It prevents "context fragmentation" where a crucial piece of information is cut off because it hit a character limit.

---

## 2. Parent-Document Retrieval (V20 Feature)
**Algorithm:** Child-to-Parent Context Mapping

### 📖 Description
This solves the **Precision vs. Context** trade-off. 
1. **Indexing:** We split a "Parent" chunk into several small, overlapping "Child" chunks.
2. **Retrieval:** We search the vector space using the small Child chunks (easier to find an exact match).
3. **Synthesis:** Once a match is found, we retrieve the broader **Parent** chunk to provide the LLM with full context.

### 🚀 Importance in RAG
Gives the LLM a much wider window of information to formulate a better, more detailed answer while keeping search speed lightning-fast.

---

## 3. HyDE (Hypothetical Document Embeddings - V20 Feature)
**Algorithm:** Zero-Shot Query Expansion

### 📖 Description
HyDE uses the LLM to generate a "fake" ideal answer to the user's question *before* searching the database. We then use the vector of this *ideal answer* to search.

### 🚀 Importance in RAG
Bridging the gap between user questions and technical documentation. It is extremely effective for short, 1-2 word queries that lack enough keywords for standard search.

---

## 4. BM25 (Best Matching 25)
**Algorithm:** Probabilistic Sparse Search (Keyword Ranking)

### 📖 Description
BM25 ranks documents based on the occurrence of query terms. It accounts for term frequency ($tf$) and inverse document frequency ($idf$).

### 🚀 Importance in RAG
Ensures that specific names, unique IDs, or rare technical terms are prioritized, even if they don't have strong "semantic" embeddings.

---

## 5. FAISS HNSW (Hierarchical Navigable Small Worlds)
**Algorithm:** Approximate Nearest Neighbor (ANN) Graph Search

### 📖 Description
Creates a multi-layered graph for vector search. It allows the engine to "jump" across the vector space to find the closest matches in milliseconds, regardless of the database size.

### 🚀 Importance in RAG
The engine's "internal GPS," allowing for near-instant retrieval across thousands of document segments.

---

## 6. Reciprocal Rank Fusion (RRF)
**Algorithm:** Rank Aggregation for Hybrid Blending

### 📖 Description
RRF combines results from Semantic Search and Keyword Search by adding their reciprocal ranks: $1 / (k + rank)$.

### 🚀 Importance in RAG
Allows the system to benefit from both "meaning" (FAISS) and "keywords" (BM25) simultaneously without needing complex score normalization.

---

## 7. Cross-Encoder Reranking
**Algorithm:** Full-Interaction Transformer Classification

### 📖 Description
A heavy-duty model that looks at the `(Question, Chunk)` pair simultaneously. It acts as a final judge to verify that the retrieved chunk actually contains the answer.

### 🚀 Importance in RAG
The "Final Filter." It eliminates irrelevant results that might have looked good to the vector search but don't actually help answer the specific question.

---

## 8. BitsAndBytes Quantization
**Algorithm:** 4-bit / 8-bit NormalFloat (NF4) Compression

### 📖 Description
Compresses the weights of the LLM from 16-bit to 4-bit, reducing the memory footprint by up to 75%.

### 🚀 Importance in RAG
Enables running professional-grade models (Llama-3, Qwen) on consumer hardware (laptops/CPUs) without needing a server-grade GPU.

---

## 9. Persistent History Architecture (V20 Feature)
**Algorithm:** SQLite Relational Persistence

### 📖 Description
Uses an asynchronous database handler to log every interaction, source cited, and performance metric to a local `.db` file.

### 🚀 Importance in RAG
Turns a temporary "session-based" script into a persistent productivity tool that remembers past context and performance trends.

---
© 2026 SourceIQ Engineering | *Documentation v20.0 — Final*

# SourceIQ – Advanced Local Multi-Document RAG Chatbot

SourceIQ is a local RAG chatbot that retrieves and answers questions from multiple documents using chunk-based retrieval and a Transformer LLM.

---

## 🚀 Features

- Multi-format ingestion support (`.txt`, `.pdf`, `.docx`, `.html`, `.htm`)
- Overlapping and Semantic Text Chunking strategies
- FAISS Vector Database Integration (HNSW index) for fast, highly scalable search
- Two-Stage Retrieval (Bi-Encoder semantic search + BM25 keyword search hybrid fusion via RRF + Cross-Encoder Reranking)
- Source tracking (file, namespace, chunk ID, and exact snippet citations)
- Conversational Memory (Alternating history and follow-up support)
- SQLite Database Integration for persistent chat history per user name
- Real-time Streaming Output (Typewriter effect in both terminal and web interfaces)
- Persistent Vector Caching (Instant startup and hot reloading)
- Transformers & GGUF (llama.cpp) local inference models (Qwen, Llama 3.2, Phi-3.5, Phi-2)
- Cloud Inference: Google Gemini API integration (Gemini 1.5 Pro / Flash)
- Sleek and Advanced Streamlit Web UI (`ui.py`) with telemetry stats and process/latencies visualization
- Interactive CLI-based Terminal chatbot (`app.py`)

---

## 🧠 How It Works

Documents → Semantic/Recursive Chunking → Vector Embeddings & Keyword Tokenization → HNSW (FAISS) + BM25 Search → Reciprocal Rank Fusion (RRF) → Cross-Encoder Reranking & Namespace Boosting → Context-grounded Answer Generation (Local LLM / GGUF / Gemini API) with Citations & SQLite History.

---

## ⚙️ Tech Stack

Python | PyTorch | Streamlit | FAISS | Hugging Face Transformers | llama-cpp-python | SQLite3 | PyMuPDF | python-docx | BS4 | Google Gemini API

---

## 🔮 Future Work

- Fully containerized Docker deployment
- Multi-user authentication system
- Advanced agentic tools (e.g. web search fallbacks)
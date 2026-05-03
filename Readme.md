# SourceIQ – Advanced Local Multi-Document RAG Chatbot

SourceIQ is a local RAG chatbot that retrieves and answers questions from multiple documents using chunk-based retrieval and a Transformer LLM.

---

## 🚀 Features

- Multi-document `.txt` ingestion
- Overlapping text chunking for better context
- Semantic Vector Embeddings (SentenceTransformers)
- Two-Stage Retrieval (Bi-Encoder search + Cross-Encoder Reranking)
- Source tracking (file + chunk ID)
- Conversational Memory (Follow-up questions)
- Real-time Streaming Output (Typewriter effect)
- Persistent Vector Caching (Instant startup)
- Context-grounded responses using 🤗 Transformers
- Qwen2.5-0.5B-Instruct model for generation
- CLI-based interactive chatbot

---

## 🧠 How It Works

Documents → Chunking → Embeddings → Bi-Encoder Search → Cross-Encoder Reranking → Chat Memory → LLM Answer

---

## ⚙️ Tech Stack

Python | PyTorch | Hugging Face Transformers | NLP | RAG

---

## 🔮 Future Work

- Web UI (Streamlit / React)  
- PDF/DOCX support  
- Fast API deployment  
- FAISS / Vector Database Integration (Scale-up)  
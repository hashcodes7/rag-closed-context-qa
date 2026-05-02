# SourceIQ – Advanced Local Multi-Document RAG Chatbot

SourceIQ is a local RAG chatbot that retrieves and answers questions from multiple documents using chunk-based retrieval and a Transformer LLM.

---

## 🚀 Features

- Multi-document `.txt` ingestion
- Overlapping text chunking for better context
- Top-K retrieval for relevant chunks
- Source tracking (file + chunk ID)
- Context-grounded responses using :contentReference[oaicite:0]{index=0} Transformers
- Qwen2.5-0.5B-Instruct model for generation
- CLI-based interactive chatbot

---

## 🧠 How It Works

Documents → Chunking → Top-K Retrieval → Context Building → LLM Answer

---

## ⚙️ Tech Stack

Python | :contentReference[oaicite:1]{index=1} | Hugging Face Transformers | NLP | RAG

---

## 🔮 Future Work

- FAISS / vector database search  
- Web UI (Streamlit / React)  
- Chat memory support  
- PDF/DOCX support  
- Fast API deployment  
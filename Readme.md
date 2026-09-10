# SourceIQ - Advanced Local Multi-Document RAG Chatbot

SourceIQ is a production-ready Retrieval-Augmented Generation (RAG) chatbot that retrieves and answers questions from multiple documents using semantic search, hybrid retrieval strategies, and advanced language models. The system supports both local inference and cloud-based APIs, with flexible deployment options via CLI or web interface.

---

## Overview

SourceIQ implements a sophisticated multi-stage retrieval and ranking pipeline to provide context-grounded responses with accurate source attribution. The system is designed for enterprise deployments, supporting multiple document formats, scalable vector databases, and persistent conversation tracking.

---

## Core Features

### Document Processing
- Multi-format ingestion support (.txt, .pdf, .docx, .html, .htm)
- Overlapping and semantic text chunking strategies
- Recursive chunking for hierarchical document understanding
- Source tracking with file name, namespace, chunk ID, and exact snippet citations

### Retrieval and Search
- FAISS Vector Database Integration (HNSW index) for scalable similarity search
- Hybrid retrieval: Bi-Encoder semantic search + BM25 keyword search
- Reciprocal Rank Fusion (RRF) for fusion of multiple ranking signals
- Cross-Encoder reranking with namespace-aware boosting
- Persistent vector caching for instant startup and hot reloading

### Inference Capabilities
- Local inference: Transformers-based models and GGUF support (llama.cpp)
- Supported local models: Qwen, Llama 3.2, Phi-3.5, Phi-2
- Cloud inference: Google Gemini API integration (Gemini 1.5 Pro / Flash)
- Context-grounded response generation with conversation memory

### User Interfaces
- Interactive CLI-based terminal chatbot (app.py)
- Sleek Streamlit web UI (ui.py) with telemetry statistics and performance visualization
- Real-time streaming output with typewriter effect
- Real-time latency and performance metrics display

### Persistence and State Management
- SQLite database integration for persistent chat history per user
- Alternating conversation history support for multi-turn interactions
- User session management and follow-up question handling

---

## How It Works

The retrieval pipeline follows this architecture:

```
Documents (Multi-format)
    |
    v
Semantic / Recursive Chunking
    |
    v
Vector Embeddings & Keyword Tokenization
    |
    v
Parallel Search: HNSW (FAISS) + BM25
    |
    v
Reciprocal Rank Fusion (RRF)
    |
    v
Cross-Encoder Reranking & Namespace Boost
    |
    v
Chat Memory Integration
    |
    v
LLM Answer Generation (Local or Cloud)
```

---

## Technology Stack

**Core Languages & Frameworks:**
- Python 3.x
- PyTorch for neural computations

**Vector and Search Infrastructure:**
- FAISS (with HNSW indexing)
- BM25 for keyword-based retrieval
- Semantic search via SentenceTransformers

**Model Inference:**
- Hugging Face Transformers
- llama-cpp-python for GGUF model support
- Google Gemini API for cloud inference

**Data and Storage:**
- SQLite3 for conversation persistence
- PyMuPDF for PDF processing
- python-docx for DOCX support
- Beautiful Soup 4 for HTML parsing

**User Interface:**
- Streamlit for web dashboard
- Interactive terminal CLI

---

## Deployment Options

### Local CLI Mode
Lightweight command-line interface for terminal-based interactions with minimal dependencies.

### Streamlit Web Interface
Full-featured web dashboard with real-time metrics, user management, and visual analytics.

### API Ready
Architecture supports FastAPI deployment for remote access and microservice integration.

---

## Roadmap

- Fully containerized Docker deployment with multi-stage builds
- Enterprise multi-user authentication and role-based access control
- Advanced agentic tools (web search fallbacks, knowledge graph integration)
- Extended document format support and preprocessing pipelines
- Performance optimizations for ultra-large document collections

---

## Installation and Usage

Refer to the branch-specific documentation for implementation details:

- **main**: Base implementation with core RAG functionality
- **without-streamlit**: Lightweight CLI-only version
- **app-specific**: Extended features with cloud API integration
- **fmc_deployment**: Production deployment configuration
- **Restrictedpc-version**: Resource-constrained environment optimizations

---

## License

a star for me on this repo would be cool. go take this code and just use it . My work is done here

---

## Support

For issues, feature requests, or contributions, please open an issue or pull request on GitHub.

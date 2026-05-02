# Comparison: Previous Code vs New Code (Advanced RAG Upgrade)

## Overview

The new code is a major upgrade over the previous RAG chatbot system.

You moved from:

- Basic multi-file RAG chatbot with line-based chunking and single-chunk retrieval

To:

- Advanced RAG pipeline with overlapping chunking, top-k retrieval, and multi-context grounding

---

# Main Differences

| Feature | Previous Code | New Code |
|--------|---------------|----------|
| Chunking Method | Line-based splitting | Overlapping word-based chunking |
| Context Structure | Flat text chunks | Structured chunks with metadata |
| Retrieval Strategy | Single best match | Top-k retrieval (k=4) |
| Context Usage | One chunk per query | Multiple chunks combined |
| Metadata Tracking | Filename only | Filename + chunk ID |
| Prompting Strategy | Basic instruction prompt | Strict grounded prompting |
| Retrieval Logic | Simple keyword match | Scored ranking system |
| Answer Quality | Limited context | Multi-context reasoning |

---

# Detailed Changes

## 1. Improved Chunking System

### Old Approach
- Each line treated as a chunk
- No context continuity between chunks
- Information loss in long documents

### New Approach
- Overlapping sliding window chunking
- Configurable chunk size and overlap

```python
def chunk_text(text, chunk_size=250, overlap=80)
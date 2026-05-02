# Comparison: Previous Code vs New Code

## Overview

The new code is an upgraded version of the previous one.

You moved from:

- Single-file local RAG chatbot using `notes.txt`

To:

- Multi-file knowledge base RAG chatbot using a dedicated folder

---

# Main Differences

| Feature | Previous Code | New Code |
|--------|---------------|----------|
| Knowledge Source | One file (`notes.txt`) | Multiple `.txt` files in folder |
| Data Loading | Single file read | Auto-load all files |
| Source Tracking | No source metadata | Tracks filename |
| Scalability | Limited | Better organized |
| Retrieval Output | Answer only | Shows source + answer |
| Knowledge Expansion | Edit one file | Add many files |
| Project Structure | Flat | Structured knowledge base |

---

# Detailed Changes

## 1. Added `os` Module

### Old Code

No filesystem directory support.

### New Code


# Comparison: Previous Code vs New Code

## Overview

The new code is an upgraded version of the previous one.

You moved from:

- File-based local knowledge chatbot (entire file passed to model)

To:

- Retrieval-based local RAG chatbot (best chunk selected before prompting)

---

# Main Differences

| Feature | Previous Code | New Code |
|--------|---------------|----------|
| Context Source | Entire `notes.txt` file | Best matching chunk from `notes.txt` |
| Retrieval System | None | Keyword overlap chunk search |
| Prompt Size | Full file context | Only selected chunk |
| Speed | Slower for large files | Faster |
| Accuracy | More noise possible | More relevant context |
| Scalability | Poor for large notes | Better |
| Debug Visibility | Only final answer | Shows best chunk + answer |
| Architecture | Zero RAG | Basic RAG |

---

# Detailed Changes

## 1. Entire File Removed From Prompt

### Old Code

```python
with open("notes.txt", "r", encoding="utf-8") as f:
    context = f.read()
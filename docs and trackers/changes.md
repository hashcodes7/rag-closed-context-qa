# Comparison: Previous Code vs New Code

## Overview

The new code is an upgraded version of the previous one.

You moved from:

- Single-question retrieval QA script

To:

- Continuous chat-based local RAG chatbot

---

# Main Differences

| Feature | Previous Code | New Code |
|--------|---------------|----------|
| Interaction Mode | One question then exits | Multi-turn chat loop |
| User Experience | Re-run script every question | Continuous conversation |
| Exit System | No built-in exit command | `quit` command added |
| Retrieval | Best chunk once | Best chunk every message |
| Output Cleaning | Printed full generated text | Clean answer extraction |
| UI | Simple input/output | Chatbot style (`You:` / `Bot:`) |
| Practical Use | QA tool | Interactive chatbot |

---

# Detailed Changes

## 1. Added Chat Loop

### Old Code

```python
question = input("Ask question: ")
# Comparison: Previous Code vs New Code

## Overview

The new code is an upgraded version of the previous one.

You moved from:

- Hardcoded paragraph question answering

To:

- File-based local knowledge chatbot

---

# Main Differences

| Feature                     | Previous Code                   | New Code                        |
|-----------------------------|---------------------------------|---------------------------------|
| Context Source              | Paragraph written inside code   | Reads from `notes.txt`          |
| Question Input              | Fixed question in code          | User types question             |
| Reusability                 | One test question               | Can ask many questions          |
| Data Size                   | Small paragraph                 | Large notes/documents           |
| Missing Answer Handling     | No fallback                     | Returns `Not found`             |
| Practical Use               | Demo/test                       | Basic chatbot                   |

---

# Detailed Changes

## 1. Hardcoded Paragraph Removed

### Old Code

```python
paragraph = """
There are myself and 4 other people...
"""
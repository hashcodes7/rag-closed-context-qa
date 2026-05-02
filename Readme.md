# Zero RAG Closed Context QA

A lightweight local question-answering chatbot built using a pretrained language model and custom prompts.

This project reads information from a local text file (`notes.txt`) and answers user questions **only from the provided context**.

It does **not** use embeddings, vector databases, retrieval pipelines, or internet access.

---

# What This Project Is

This project demonstrates a simple alternative to traditional RAG systems.

Instead of:

- Storing vectors
- Searching embeddings
- Retrieving chunks

It directly loads text from a file and injects it into the model prompt.

This approach can be called:

- Zero RAG
- Closed Context QA
- Prompt-based Local QA
- Manual Context Injection

---

# Features

- Uses local text file as knowledge source
- Runs locally on your machine
- Works offline after first model download
- User can ask custom questions
- Responds only from provided notes
- Returns `Not found` when answer is missing
- Lightweight and beginner-friendly

---

# Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Qwen2.5-0.5B-Instruct

---

# Recommended Environment

## Operating System

- Windows 10 / 11
- Linux
- macOS

## Python Version

```text id="e0nyqf"
Python 3.10+
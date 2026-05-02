# README — Local LLM Question Answering Demo (Not True RAG)

## Project Overview

This project demonstrates how to run a **local language model** using Hugging Face Transformers and ask questions based on custom text.

It uses the **Qwen2.5-0.5B-Instruct** model to read a paragraph and generate answers.

---

# Important Clarification

## This is **NOT** a True RAG System

Many beginners think that giving text to a model inside a prompt means they built RAG.

That is **not correct**.

This project currently uses **Prompt Injection / Context Prompting**, not Retrieval-Augmented Generation (RAG).

---

# What This Code Actually Does

The paragraph is manually inserted into the prompt like this:

```python
prompt = f"""
Paragraph:
{paragraph}

Question:
{question}

Answer:
"""
# 🤖 V3 RAG-style Closed Context QA Bot

## 🏗️ Summary

> [!NOTE]
> **Goal For This Version**  
> Build a **V3 RAG-style Closed Context QA Bot**. It implements a true **Retrieval-Augmented Generation (RAG)** pipeline by introducing **document chunking** and a basic **keyword-based retrieval system** to feed only the most relevant paragraph into the language model.

> [!IMPORTANT]
> **Changes from Last Version [v2] to Current Version [v3]**  
> * Introduced a chunking mechanism (`split("\n")`) to break the document into smaller pieces.
> * Added a keyword-overlap scoring loop to find the single most relevant chunk.
> * The prompt now receives only the `best_chunk` instead of the entire document.
> * Prints out the retrieved `best_chunk` before generating the final answer.

> [!IMPORTANT]
> **Why the changes were made (problem faced)**  
> Imagine you are a student sitting for an exam, and your teacher hands you an enormous 500-page textbook and says: *"Your answer to this question must come from somewhere in here."* You'd be overwhelmed, right? That is exactly what we were doing to the AI model in V2.
>
> In V2, the **entire** `notes.txt` file was stuffed directly into the model's prompt as raw text before every single question. While this feels intuitive — just give the AI everything and let it figure it out — it runs into a very real hardware limitation called the **context window**.
>
> A language model's context window is like its working memory. It can only hold and process a certain number of "tokens" (roughly, pieces of words) at one time. The Qwen 0.5B model we are using has a relatively small context window. When you throw a very large document at it, two terrible things happen. First, the model might simply crash or throw an out-of-memory error because the text is too long to fit. Second, even if it doesn't crash, the model gets genuinely confused — it has to "pay attention" to hundreds of paragraphs at once, and important information gets diluted in the noise. The answer quality drops significantly because the model struggles to find the relevant needle in that massive haystack.
>
> Think of it like asking someone to remember everything on every page of a book you just read out loud to them, then asking one small question. Even a human would struggle.

> [!IMPORTANT]
> **How the new version solves the problem**  
> V3 introduces the most fundamental idea in all of RAG: **don't give the model everything — give it only what it needs.**
>
> Here's the approach, step by step in plain English. First, we take the big document and cut it into many small, bite-sized pieces called **chunks**. Think of it like cutting a long pizza into individual slices. Each slice is a paragraph. We then look at every single slice and ask: *"How many words in this paragraph match the words in the user's question?"* We count those matching words and give each paragraph a score. The paragraph with the highest score — the one that shares the most vocabulary with the question — wins. We call this the `best_chunk`.
>
> Now here's the magic: instead of feeding the AI the whole pizza box, we hand it only that one winning slice. The prompt becomes tiny, laser-focused, and completely within the model's memory limits. The AI doesn't have to search through irrelevant paragraphs about something unrelated — it reads one small, relevant piece of text and gives a sharp, direct answer.
>
> This technique of splitting → scoring → retrieving is the **core concept of Retrieval-Augmented Generation (RAG)**, and it forms the foundation that every future version of this bot will build upon.

---

## 📖 Terminologies

| Term | What It Means |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | A technique where the AI first *retrieves* relevant information from a database, then *generates* an answer based only on that retrieved information. It prevents the model from making things up. |
| **Context Window** | The maximum amount of text a language model can read and process at one time. Think of it as the model's short-term memory — it can only hold so much before it starts forgetting or getting confused. |
| **Chunk** | A small, self-contained piece of a larger document. Instead of feeding the whole document, we break it into chunks so we can pick and choose only the most relevant one. |
| **Keyword Overlap** | A simple scoring method: count how many words from the user's question appear in a given chunk. The chunk with the most matching words gets the highest score. |
| **Prompt** | The full text input that gets sent to the AI model. It includes instructions, context, and the question. The model reads the prompt and generates its answer from it. |
| **Tokenizer** | A tool that converts human-readable text into numbers (tokens) that the AI model can process. It also converts the model's number output back into readable text. |
| **Token** | A small unit of text — roughly a word or part of a word. Models count their memory usage in tokens, not characters or words. |
| **`best_chunk`** | The single paragraph from the document that scored highest based on keyword overlap with the user's question. Only this chunk gets passed to the AI. |

---

## 🏗️ Architecture

### 🔄 Full System Flow

```mermaid
flowchart TD
    A[📄 notes.txt] -->|Load Text| B[🪓 Chunking System]
    B -->|List of Chunks| C[🔍 Keyword Retrieval]
    D[👤 Interactive Question] -->|Input| C
    D -->|User Question| E[🛠️ Prompt Builder]
    C -->|Best Chunk| E
    E -->|Final Prompt| F[🔠 Tokenizer]
    F -->|Tokens| G[🧠 Qwen Model]
    G -->|Generation| H[🔢 Generated Tokens]
    H -->|Decode| I[🗣️ Answer Text]
    I --> J[🎯 Output Shown]

    classDef file fill:#e1f5fe,stroke:#01579b;
    classDef user fill:#fff3e0,stroke:#e65100;
    classDef core fill:#e8f5e9,stroke:#1b5e20;
    classDef new_logic fill:#f3e5f5,stroke:#4a148c;

    class A file;
    class D user;
    class E,F,G core;
    class B,C new_logic;
```

---

### 📦 Code

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True
)

# Read file
with open("notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 🆕 NEW IN V3: Split into paragraph chunks
chunks = [c.strip() for c in text.split("\n") if c.strip()]

question = input("Ask question: ").lower()

# 🆕 NEW IN V3: Score chunks by keyword overlap to find the best match
best_chunk = ""
best_score = -1

for chunk in chunks:
    score = 0
    for word in question.split():
        if word in chunk.lower():
            score += 1

    if score > best_score:
        best_score = score
        best_chunk = chunk

# 🆕 NEW IN V3: Inject only the best_chunk into the prompt
prompt = f"""
Answer only using the context below.
If answer not found, say Not found.

Context:
{best_chunk}

Question:
{question}

Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🆕 NEW IN V3: Display the retrieved context for transparency
print("\nBEST CHUNK:\n")
print(best_chunk)

print("\nRESULT:\n")
print(answer)
```

---

## 🏗️ Stepwise Architecture

### 📦 Step 1 — Import Required Libraries

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
```

> [!TIP]
> **Purpose:**
> * `transformers`: Loads the tokenizer and model.
> * `torch`: Runs inference operations.

---

### 🧠 Step 2 — Define Model

```python
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
```

**Purpose:** Selects the language model used for answering.

---

### 🔠 Step 3 — Load Tokenizer

```python
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

**Purpose:** Tokenizer converts human text into tokens the model understands.

---

### 📥 Step 4 — Load Model

```python
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True
)
```

**Purpose:** Loads the neural network weights into memory efficiently.

---

### 📚 Step 5 — Load External Knowledge Source

```python
with open("notes.txt", "r", encoding="utf-8") as f:
    text = f.read()
```

**Purpose:** Reads the raw text from the external `notes.txt` file.

---

### 🪓 Step 6 — Split Text Into Chunks (🆕 NEW IN V3)

```python
chunks = [c.strip() for c in text.split("\n") if c.strip()]
```

**Purpose:** Splits the massive block of text into smaller, individual paragraphs using newline characters (`\n`). This prepares the data so we only search for relevant pieces instead of reading everything at once.

---

### ❓ Step 7 — Accept Interactive User Question

```python
question = input("Ask question: ").lower()
```

**Purpose:** Pauses execution to take the user's question, converting it to lowercase for easier text matching later.

---

### 🔍 Step 8 — Retrieve Best Chunk via Keyword Overlap (🆕 NEW IN V3)

```python
best_chunk = ""
best_score = -1

for chunk in chunks:
    score = 0
    for word in question.split():
        if word in chunk.lower():
            score += 1

    if score > best_score:
        best_score = score
        best_chunk = chunk
```

**Purpose:** Loops through every paragraph. It scores each chunk by counting how many words from the user's question appear in it. The paragraph with the highest score is saved as `best_chunk`.

---

### 📝 Step 9 — Build Prompt with Best Chunk (🆕 NEW IN V3)

```python
prompt = f"""
Answer only using the context below.
If answer not found, say Not found.

Context:
{best_chunk}

Question:
{question}

Answer:
"""
```

**Purpose:** Combines rules, the user's question, and **only the highest-scoring chunk** (`best_chunk`). This protects the model from reading irrelevant data and saves token memory.

---

### 🔢 Step 10 — Convert Prompt to Tokens

```python
inputs = tokenizer(prompt, return_tensors="pt")
```

**Purpose:** Transforms prompt text into tensors for PyTorch.

---

### ⚙️ Step 11 — Generate Answer

```python
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )
```

**Purpose:** Model reads the prompt and generates a response based exclusively on the narrowed-down chunk.

---

### 🗣️ Step 12 — Decode Output

```python
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Purpose:** Converts generated tokens back into readable text.

---

### 🖨️ Step 13 — Print Context & Final Result (🆕 NEW IN V3)

```python
print("\nBEST CHUNK:\n")
print(best_chunk)

print("\nRESULT:\n")
print(answer)
```

**Purpose:** Prints the retrieved `best_chunk` so the user can see exactly what text the AI read, followed by the final generated answer.

---

## 🚀 Final One-Line Understanding

> **This architecture splits a file into pieces, uses basic word-matching to find the best piece, and feeds only that specific piece to the AI to answer the user's question.**

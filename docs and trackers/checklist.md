# Core Build Steps Progress Tracker

## Step 1 — Local Model Works (v1)
* [x] Create `app.py`
* [x] Load tokenizer and Qwen model
* [x] Ask one hardcoded question against hardcoded text
* [x] Print answer in terminal

## Step 2 — TXT Knowledge Source (v2)
* [x] Read text from external `notes.txt` file
* [x] Accept dynamic user questions via `input()`

## Step 3 — Basic RAG Logic (v3)
* [x] Split text into chunks using `\n`
* [x] Find the single best chunk via exact keyword overlap
* [x] Pass only the best chunk to the model to save tokens

## Step 4 — Chatbot Mode (v4)
* [x] Implement continuous `while True:` chat loop
* [x] Add graceful `quit` exit condition
* [x] Clean AI output by extracting text after "Answer:"

## Step 5 — Multi-Document Citations (v5)
* [x] Scan `knowledge_source` directory for multiple `.txt` files
* [x] Store chunks as dictionaries with source filename metadata
* [x] Print exact file citation before generating answer

## Step 6 — Advanced Structure & Telemetry (v6)
* [x] Implement sliding window chunking (250 words, 80 overlap)
* [x] Store `chunk_id` in metadata
* [x] Retrieve Top-K (4) chunks instead of a single chunk
* [x] Combine multiple chunks into a metadata-tagged context string
* [x] Upgrade to Strict System Prompt to prevent hallucinations
* [x] Add `time.time()` telemetry logging for backend metrics

## Step 6.1 — Context Truncation Patch (v6.1)
* [x] Add `truncate()` function to cap chunks at 120 words
* [x] Simplify metadata tags in context builder
* [x] Lower `max_new_tokens` to 60 for faster, punchier answers

## Step 7 — Semantic Search & Embeddings (v7)
* [x] Import `sentence-transformers` embedding model
* [x] Convert document chunks into mathematical vector matrices on startup
* [x] Convert user query into a mathematical vector
* [x] Replace word-counting with PyTorch `cosine_similarity`
* [x] Retrieve top-k conceptually matching chunks regardless of exact vocabulary

---

# 🚀 Upcoming Upgrade Roadmap

## Step 8 — Conversational Memory (Chat History) (v8)
* [x] Initialize a rolling chat history buffer
* [x] Append previous user inputs and bot answers to memory
* [x] Inject the last 2-3 exchanges into the prompt for conversational context

## Step 8.1 — Memory Patch for 0.5B Models (v8.1)
* [x] Isolate chat history and context using XML tags `<chat_history>` and `<context>`
* [x] Truncate bot's past answers to 100 chars to prevent parrot looping
* [x] Simplify negative response rules to strictly "Not found."

## Step 9 — Two-Stage Retrieval (Reranking) (v9)
* [x] Introduce a Cross-Encoder model
* [x] Retrieve top 10 chunks via fast Bi-Encoder (V7)
* [x] Re-score the 10 chunks using the highly accurate Cross-Encoder
* [x] Pass only the absolute best 3 chunks to the generative model

## Step 9.1 — ChatML Instruction Patch (v9.1)
* [x] Delete raw f-string prompt formatting
* [x] Structure prompt as a list of `role` dictionaries
* [x] Inject chat history as literal `user` and `assistant` messages
* [x] Use `tokenizer.apply_chat_template()` to compile native instruction tokens
* [x] Slice model generation output via exact tensor length extraction

## Step 10 — Streaming Output (Typewriter Effect) (v10)
* [ ] Implement Hugging Face `TextStreamer`
* [ ] Bypass wait time by printing tokens to console in real-time
* [ ] Create a ChatGPT-like fluid UI experience

## Step 11 — Persistent Vector Caching (v11)
* [ ] Integrate FAISS or local disk serialization (`.pt` files)
* [ ] Save generated chunk embeddings and metadata to local disk
* [ ] Load pre-computed database instantly on startup to bypass embedding wait times

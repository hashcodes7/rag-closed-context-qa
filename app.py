from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import faiss
import numpy as np
import os
import time
from threading import Thread

# =====================================================
# 🧠 RAGBOT V14
# Level 8 — Model Quantization (Performance)
# 4-bit/8-bit loading via bitsandbytes to save VRAM
# =====================================================

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# -----------------------------------------------------
# 🟢 CONFIGURATION
# -----------------------------------------------------
# 🆕 NEW IN V14: Choose quantization mode ("4bit", "8bit", "full")
# "4bit" is recommended for lowest VRAM usage with minimal quality loss.
QUANTIZATION_MODE = "4bit"

print(f"🔄 RAGBOT V14 Running (Mode: {QUANTIZATION_MODE})........")

print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ Tokenizer loaded")

print(f"🔄 Loading Qwen model ({QUANTIZATION_MODE})...")

# 🆕 NEW IN V14: Configure BitsAndBytes for quantization
quantization_config = None
if QUANTIZATION_MODE == "4bit":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
elif QUANTIZATION_MODE == "8bit":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map="auto" if quantization_config else None
    )
    print(f"✅ Qwen model loaded in {QUANTIZATION_MODE} mode")
except Exception as e:
    print(f"⚠️ Quantization failed: {e}")
    print("🔄 Falling back to Full Precision loading...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    )
    print("✅ Qwen model loaded (Full Precision)")

print("🔄 Loading embedding model (Bi-Encoder)...")
embedder = SentenceTransformer(embed_model_name)
print("✅ Embedding model loaded")

# 🆕 NEW IN V9: Load Cross-Encoder for reranking
print("🔄 Loading reranker model (Cross-Encoder)...")
cross_encoder = CrossEncoder(cross_encoder_model_name)
print("✅ Reranker model loaded")


# -----------------------------------------------------
# 🟢 CHUNKING
# -----------------------------------------------------

# 🆕 NEW IN V12: Sentence-aware Recursive Character Splitter
def recursive_chunk_text(text, chunk_size=1000, overlap=200):
    """
    Splits text recursively using a separator hierarchy:
      1. Paragraph breaks (\n\n)
      2. Line breaks (\n)
      3. Sentence endings ('. ')
      4. Word boundaries (' ')
    This prevents facts from being clipped mid-sentence.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text, separators):
        """Recursively split text using the next separator in the hierarchy."""
        if not text.strip():
            return []

        # If the text is already small enough, return it as-is
        if len(text) <= chunk_size:
            return [text.strip()]

        sep = separators[0]
        remaining_seps = separators[1:]

        parts = text.split(sep)

        chunks = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                # Flush the current buffer
                if current.strip():
                    if len(current) > chunk_size and remaining_seps:
                        # Current buffer is still too large — recurse with finer separator
                        chunks.extend(_split(current, remaining_seps))
                    else:
                        chunks.append(current.strip())

                # Start a new buffer with overlap from the last chunk
                if chunks and overlap > 0:
                    last = chunks[-1]
                    overlap_text = last[-overlap:].strip()
                    current = (overlap_text + " " + part.strip()).strip()
                else:
                    current = part.strip()

        # Flush any remaining buffer
        if current.strip():
            if len(current) > chunk_size and remaining_seps:
                chunks.extend(_split(current, remaining_seps))
            else:
                chunks.append(current.strip())

        return chunks

    return _split(text, separators)


def truncate(text, max_words=120):
    return " ".join(text.split()[:max_words])


# -----------------------------------------------------
# 🟠 LOADING & CACHING (V13 — Dual-File FAISS Cache)
# -----------------------------------------------------

CACHE_FILE = "vector_cache.pt"       # stores chunk metadata (source, text)
FAISS_INDEX_FILE = "faiss_index.bin" # stores the FAISS HNSW index
chunks = []
faiss_index = None

# 🆕 NEW IN V13: Both cache files must exist together for a valid cache hit
if os.path.exists(CACHE_FILE) and os.path.exists(FAISS_INDEX_FILE):
    print(f"\n💾 Loading cached knowledge base from {CACHE_FILE}...")
    start = time.time()
    chunks = torch.load(CACHE_FILE)["chunks"]
    print(f"✅ Chunks loaded: {len(chunks)}")

    print(f"⚡ Loading FAISS index from {FAISS_INDEX_FILE}...")
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"✅ FAISS index loaded in {time.time() - start:.2f}s ({faiss_index.ntotal} vectors)")
else:
    print("\n📂 No cache found. Processing knowledge base from scratch...")

    # --- LOAD KNOWLEDGE BASE ---
    folder = "knowledge_source"
    file_count = 0
    for filename in os.listdir(folder):
        if not filename.endswith(".txt"): continue
        file_count += 1
        print(f"📄 Reading file: {filename}")
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"✂ Chunking file: {filename}")
        text_chunks = recursive_chunk_text(text)
        for i, chunk in enumerate(text_chunks):
            chunks.append({"source": filename, "chunk_id": i, "text": chunk})

    print(f"✅ Loaded {len(chunks)} chunks from {file_count} files")

    # --- GENERATE EMBEDDINGS ---
    print("\n🧠 Creating embeddings for all chunks...")
    chunk_texts = [item["text"] for item in chunks]
    start = time.time()
    embeddings_np = embedder.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    print(f"✅ Chunk embeddings ready in {time.time() - start:.2f}s")

    # --- BUILD FAISS HNSW INDEX ---
    print("\n🏗️ Building FAISS HNSW index...")
    index_start = time.time()

    # 🆕 NEW IN V13: L2-normalize so inner product == cosine similarity
    faiss.normalize_L2(embeddings_np)

    dim = embeddings_np.shape[1]
    faiss_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    faiss_index.hnsw.efConstruction = 200  # Build-time graph accuracy
    faiss_index.add(embeddings_np)

    print(f"✅ HNSW index built in {time.time() - index_start:.2f}s ({faiss_index.ntotal} vectors indexed)")

    # --- SAVE DUAL CACHE ---
    print(f"\n💾 Saving chunks to {CACHE_FILE}...")
    torch.save({"chunks": chunks}, CACHE_FILE)

    print(f"💾 Saving FAISS index to {FAISS_INDEX_FILE}...")
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    print("✅ Both cache files saved successfully")


# -----------------------------------------------------
# 🟠 SEMANTIC RETRIEVAL (V13 FAISS HNSW + Cross-Encoder)
# -----------------------------------------------------
def retrieve_top_k(question, k=3):
    print("\n🔍 Stage 1: FAISS HNSW Search (Bi-Encoder)...")
    start = time.time()

    # 🆕 NEW IN V13: Encode query as float32 numpy array and normalize
    query_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)  # Must match how the index was built

    # Set query-time accuracy (higher = more accurate, slower)
    faiss_index.hnsw.efSearch = 64

    # Search the HNSW index — returns (distances, indices) as 2D arrays
    distances, indices = faiss_index.search(query_vec, k=min(10, faiss_index.ntotal))

    initial_results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:  # FAISS pads with -1 when k > ntotal
            continue
        item = chunks[idx].copy()
        initial_results.append(item)

    print(f"✅ Stage 1 retrieved {len(initial_results)} broad chunks in {time.time() - start:.2f}s")

    # Stage 2 — Cross-Encoder Reranking (unchanged from V9)
    print("🎯 Stage 2: Cross-Encoder Reranking...")
    rerank_start = time.time()

    cross_inp = [[question, item["text"]] for item in initial_results]
    cross_scores = cross_encoder.predict(cross_inp)

    # Attach new scores and sort
    for i in range(len(initial_results)):
        initial_results[i]["score"] = float(cross_scores[i])

    initial_results.sort(key=lambda x: x["score"], reverse=True)

    # Select the absolute best 'k' chunks
    final_results = initial_results[:k]

    print(f"✅ Stage 2 reranked and selected top {k} chunks in {time.time() - rerank_start:.2f}s")

    return final_results


# -----------------------------------------------------
# 🟢 CHAT LOOP
# -----------------------------------------------------

print(f"\n🤖 RAGBOT V14 Ready! (Mode: {QUANTIZATION_MODE}) Type 'quit' to exit.\n")

# 🆕 NEW IN V8: Initialize rolling chat history buffer
chat_history = []

while True:
    question = input("You: ").strip()

    if question.lower() == "quit":
        print("👋 Goodbye.")
        break

    if not question:
        continue

    print("\n==============================")
    print("📥 QUESTION RECEIVED")
    print("==============================")
    print("🧾 Question:", question)

    # ---------------------------------------------
    # RETRIEVE TOP CHUNKS
    # ---------------------------------------------

    top_chunks = retrieve_top_k(question, k=3)

    if not top_chunks:
        print("❌ No relevant context found.\n")
        continue

    # ---------------------------------------------
    # BUILD CONTEXT
    # ---------------------------------------------

    print("\n🧱 Building context...")

    context = ""
    sources = set()

    for c in top_chunks:
        context += (
            f"\n[Source: {c['source']} | Score: {c['score']:.3f}]\n"
            f"{truncate(c['text'])}\n"
        )
        sources.add(c["source"])

    print(f"📚 Context size: {len(context)} characters")

    # ---------------------------------------------
    # PROMPT (V9.1 ChatML Patch)
    # ---------------------------------------------

    print("\n🧠 Creating prompt...")

    # 🆕 NEW IN V9.1: Use native ChatML message structuring instead of raw f-strings
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an assistant. Answer the user's question using ONLY the provided context.\n"
                f"<context>\n{context}\n</context>\n"
                "If the answer is not in the context, reply exactly with 'Not found.' Do not add explanations."
            )
        }
    ]

    # Inject the last 2 conversational turns as actual chat history messages
    for entry in chat_history[-2:]:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["bot"]})

    # Add the current question
    messages.append({"role": "user", "content": question})

    # Automatically format the messages using Qwen's native tags (<|im_start|>)
    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # ---------------------------------------------
    # TOKENIZE
    # ---------------------------------------------

    print("\n🔤 Tokenizing input...")

    inputs = tokenizer(text_prompt, return_tensors="pt")

    print(f"🧮 Input tokens: {inputs['input_ids'].shape[1]}")

    # ---------------------------------------------
    # GENERATE (V10 Streaming Upgrade)
    # ---------------------------------------------

    print("\n⚙ Generating response...")

    # Setup the streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=150, # Increased for more descriptive streaming
        do_sample=False
    )

    # Start generation in a separate thread so we can iterate the streamer in main
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    
    print("\n==============================")
    print("📌 FINAL RESULT")
    print("==============================")
    print("\n📁 Sources:", ", ".join(sources))
    print("\n🤖 Bot: ", end="", flush=True)

    start_time = time.time()
    thread.start()

    full_response = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        full_response += new_text

    print(f"\n\n✅ Generation done in {time.time() - start_time:.2f}s\n")

    # 🆕 NEW IN V8/V10: Append current exchange to history
    chat_history.append({
        "user": question,
        "bot": full_response.strip()
    })
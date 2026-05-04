from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import faiss
import numpy as np
import os
import time
import math
from threading import Thread

# =====================================================
# 🧠 RAGBOT V15
# Level 9 — Hybrid Search (Performance)
# Semantic + BM25 Keyword Search combined with RRF
# =====================================================

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# -----------------------------------------------------
# 🟢 CONFIGURATION
# -----------------------------------------------------
# 🆕 NEW IN V15: Hybrid Search toggle and RRF constant
USE_HYBRID = True
RRF_K = 60

# 🆕 NEW IN V14: Choose quantization mode ("4bit", "8bit", "full")
QUANTIZATION_MODE = "4bit"

print(f"🔄 RAGBOT V15 Running (Hybrid: {USE_HYBRID}, Mode: {QUANTIZATION_MODE})........")

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
# 🟢 UTILITIES
# -----------------------------------------------------

# 🆕 NEW IN V15: Simple BM25 implementation for keyword search
class SimpleBM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        
        nd = {} # word -> number of docs containing word
        for doc in corpus:
            self.doc_len.append(len(doc))
            frequencies = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1
        
        for word, freq in nd.items():
            # Standard BM25 IDF formula
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query):
        scores = [0.0] * self.corpus_size
        for word in query:
            if word not in self.idf: continue
            idf = self.idf[word]
            for i in range(self.corpus_size):
                fi = self.doc_freqs[i].get(word, 0)
                # BM25 score formula for this term
                scores[i] += idf * (fi * (self.k1 + 1)) / (fi + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
        return scores

def tokenize(text):
    # Simple whitespace/lowercase tokenization
    return text.lower().replace(".", " ").replace(",", " ").replace("?", " ").split()

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
bm25_index = None

# 🆕 NEW IN V15: Reciprocal Rank Fusion (RRF) implementation
def reciprocal_rank_fusion(results_list, k=60):
    """
    Combines multiple ranked lists into one using RRF.
    results_list: list of lists containing chunk indices.
    """
    fused_scores = {}
    for results in results_list:
        for rank, idx in enumerate(results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank)
    
    # Sort by fused score descending
    sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return sorted_indices

# 🆕 NEW IN V13: Both cache files must exist together for a valid cache hit
if os.path.exists(CACHE_FILE) and os.path.exists(FAISS_INDEX_FILE):
    print(f"\n💾 Loading cached knowledge base from {CACHE_FILE}...")
    start = time.time()
    chunks = torch.load(CACHE_FILE)["chunks"]
    print(f"✅ Chunks loaded: {len(chunks)}")

    print(f"⚡ Loading FAISS index from {FAISS_INDEX_FILE}...")
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"✅ FAISS index loaded in {time.time() - start:.2f}s ({faiss_index.ntotal} vectors)")

    # 🆕 NEW IN V15: Initialize BM25 index from loaded chunks
    print("⚡ Initializing BM25 index...")
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25_index = SimpleBM25(tokenized_corpus)
    print("✅ BM25 index ready")
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

    # 🆕 NEW IN V15: Initialize BM25 index after processing from scratch
    print("🏗️ Initializing BM25 index...")
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25_index = SimpleBM25(tokenized_corpus)
    print("✅ BM25 index ready")


# -----------------------------------------------------
# 🟠 SEMANTIC RETRIEVAL (V13 FAISS HNSW + Cross-Encoder)
# -----------------------------------------------------
def retrieve_top_k(question, k=3):
    # Stage 1a — Semantic Search (FAISS HNSW)
    semantic_start = time.time()
    query_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    faiss_index.hnsw.efSearch = 64
    s_dist, s_indices = faiss_index.search(query_vec, k=min(20, faiss_index.ntotal))
    semantic_ids = [int(idx) for idx in s_indices[0] if idx != -1]
    print(f"✅ Stage 1a (Semantic) found {len(semantic_ids)} chunks in {time.time() - semantic_start:.4f}s")

    # 🆕 NEW IN V15: Stage 1b — Keyword Search (BM25)
    keyword_start = time.time()
    tokenized_query = tokenize(question)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    # Get top 20 indices based on BM25 scores
    keyword_ids = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
    print(f"✅ Stage 1b (Keyword) found {len(keyword_ids)} chunks in {time.time() - keyword_start:.4f}s")

    # 🆕 NEW IN V15: Hybrid Fusion using RRF
    fusion_start = time.time()
    fused_ids = reciprocal_rank_fusion([semantic_ids, keyword_ids], k=RRF_K)
    # Take top 10 for reranking
    initial_results = [chunks[idx].copy() for idx in fused_ids[:10]]
    print(f"✅ Stage 1c (RRF Fusion) merged into {len(initial_results)} candidates in {time.time() - fusion_start:.4f}s")

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

print(f"\n🤖 RAGBOT V15 Ready! (Hybrid: {USE_HYBRID}, Mode: {QUANTIZATION_MODE}) Type 'quit' to exit.\n")

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
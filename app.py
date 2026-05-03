from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import torch.nn.functional as F
import os
import time
from threading import Thread

# =====================================================
# 🧠 RAGBOT V11 (The Finale)
# Level 5 — Persistent Vector Caching
# Keeps same terminal style + same Qwen model
# =====================================================

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# -----------------------------------------------------
# 🟢 LOADING PHASE
# -----------------------------------------------------

print("🔄 RAGBOT V11 Running........")

print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ Tokenizer loaded")

print("🔄 Loading Qwen model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True
)
print("✅ Qwen model loaded")

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

def chunk_text(text, chunk_size=250, overlap=80):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def truncate(text, max_words=120):
    return " ".join(text.split()[:max_words])


# -----------------------------------------------------
# 🟠 LOADING & CACHING (V11 Final Upgrade)
# -----------------------------------------------------

CACHE_FILE = "vector_cache.pt"
chunks = []
chunk_embeddings = None

if os.path.exists(CACHE_FILE):
    print(f"\n💾 Loading cached knowledge base from {CACHE_FILE}...")
    start = time.time()
    cache_data = torch.load(CACHE_FILE)
    chunks = cache_data["chunks"]
    chunk_embeddings = cache_data["embeddings"]
    print(f"✅ Cache loaded in {time.time() - start:.2f}s ({len(chunks)} chunks)")
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
        text_chunks = chunk_text(text)
        for i, chunk in enumerate(text_chunks):
            chunks.append({"source": filename, "chunk_id": i, "text": chunk})
    
    print(f"✅ Loaded {len(chunks)} chunks from {file_count} files")

    # --- GENERATE EMBEDDINGS ---
    print("\n🧠 Creating embeddings for all chunks...")
    chunk_texts = [item["text"] for item in chunks]
    start = time.time()
    chunk_embeddings = embedder.encode(chunk_texts, convert_to_tensor=True, show_progress_bar=True)
    print(f"✅ Chunk embeddings ready in {time.time() - start:.2f}s")

    # --- SAVE TO CACHE ---
    print(f"💾 Saving to cache: {CACHE_FILE}...")
    torch.save({"chunks": chunks, "embeddings": chunk_embeddings}, CACHE_FILE)
    print("✅ Cache saved successfully")


# -----------------------------------------------------
# 🟠 SEMANTIC RETRIEVAL (V9 TWO-STAGE)
# -----------------------------------------------------
def retrieve_top_k(question, k=3):
    print("\n🔍 Stage 1: Fast Semantic Retrieval (Bi-Encoder)...")
    start = time.time()

    query_embedding = embedder.encode(
        question,
        convert_to_tensor=True
    )

    scores = F.cosine_similarity(
        query_embedding.unsqueeze(0),
        chunk_embeddings
    )

    # 🆕 NEW IN V9: Retrieve top 10 chunks initially
    top_scores, top_indices = torch.topk(scores, k=min(10, len(chunks)))

    initial_results = []
    for score, idx in zip(top_scores, top_indices):
        item = chunks[idx.item()].copy()
        initial_results.append(item)

    print(f"✅ Stage 1 retrieved {len(initial_results)} broad chunks in {time.time() - start:.2f}s")

    # 🆕 NEW IN V9: Stage 2 - Cross-Encoder Reranking
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

print("\n🤖 RAGBOT V11 Ready! Type 'quit' to exit.\n")

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
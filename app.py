from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import os
import time

# =====================================================
# 🧠 RAGBOT V8.1 (Patch)
# Level 2 — Semantic Retrieval + Conversational Memory
# Keeps same terminal style + same Qwen model
# =====================================================

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------------------------------
# 🟢 LOADING PHASE
# -----------------------------------------------------

print("🔄 RAGBOT V8.1 Running........")

print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ Tokenizer loaded")

print("🔄 Loading Qwen model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True
)
print("✅ Qwen model loaded")

print("🔄 Loading embedding model...")
embedder = SentenceTransformer(embed_model_name)
print("✅ Embedding model loaded")


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
# 🟢 LOAD KNOWLEDGE BASE
# -----------------------------------------------------

print("\n📂 Loading knowledge base...")

chunks = []
folder = "knowledge_source"
file_count = 0

for filename in os.listdir(folder):
    if not filename.endswith(".txt"):
        continue

    file_count += 1
    print(f"📄 Reading file: {filename}")

    path = os.path.join(folder, filename)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"✂ Chunking file: {filename}")
    text_chunks = chunk_text(text)

    for i, chunk in enumerate(text_chunks):
        chunks.append({
            "source": filename,
            "chunk_id": i,
            "text": chunk
        })

print(f"✅ Loaded {len(chunks)} chunks from {file_count} files")


# -----------------------------------------------------
# 🟠 LEVEL 2 — GENERATE CHUNK EMBEDDINGS
# -----------------------------------------------------

print("\n🧠 Creating embeddings for all chunks...")

chunk_texts = [item["text"] for item in chunks]

start = time.time()

chunk_embeddings = embedder.encode(
    chunk_texts,
    convert_to_tensor=True,
    show_progress_bar=True
)

print(f"✅ Chunk embeddings ready in {time.time() - start:.2f}s")


# -----------------------------------------------------
# 🟠 SEMANTIC RETRIEVAL
# -----------------------------------------------------

def retrieve_top_k(question, k=3):
    print("\n🔍 Semantic retrieval started...")

    start = time.time()

    # Embed query
    query_embedding = embedder.encode(
        question,
        convert_to_tensor=True
    )

    # Cosine similarity
    scores = F.cosine_similarity(
        query_embedding.unsqueeze(0),
        chunk_embeddings
    )

    # Top K indexes
    top_scores, top_indices = torch.topk(scores, k=min(k, len(chunks)))

    results = []

    for score, idx in zip(top_scores, top_indices):
        item = chunks[idx.item()].copy()
        item["score"] = float(score.item())
        results.append(item)

    print(f"✅ Retrieved {len(results)} chunks in {time.time() - start:.2f}s")

    return results


# -----------------------------------------------------
# 🟢 CHAT LOOP
# -----------------------------------------------------

print("\n🤖 RAGBOT V8.1 Ready! Type 'quit' to exit.\n")

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

    # 🆕 NEW IN V8.1: Format history with Q/A to prevent parrot looping
    history_text = "No previous history."
    if chat_history:
        history_text = ""
        # Keep only the last 2 exchanges to prevent token bloat and confusion
        for entry in chat_history[-2:]:
            # Truncate long bot answers so the model doesn't fixate on them
            short_bot = entry['bot'][:100] + "..." if len(entry['bot']) > 100 else entry['bot']
            history_text += f"User: {entry['user']}\nAssistant: {short_bot}\n"

    # ---------------------------------------------
    # PROMPT
    # ---------------------------------------------

    print("\n🧠 Creating prompt...")

    # 🆕 NEW IN V8.1: Simplified rules and added <xml> tags to help 0.5B model parse data
    prompt = f"""
You are an assistant. Answer the Current Question using ONLY the <context>.
If the answer is not in the <context>, reply exactly with "Not found."
Use <chat_history> only to understand pronouns or references in the Current Question. Do not repeat the chat history.

<chat_history>
{history_text}
</chat_history>

<context>
{context}
</context>

Current Question: {question}

Answer:
"""

    print(f"📏 Prompt length: {len(prompt)} characters")

    # ---------------------------------------------
    # TOKENIZE
    # ---------------------------------------------

    print("\n🔤 Tokenizing input...")

    inputs = tokenizer(prompt, return_tensors="pt")

    print(f"🧮 Input tokens: {inputs['input_ids'].shape[1]}")

    # ---------------------------------------------
    # GENERATE
    # ---------------------------------------------

    print("\n⚙ Generating response...")

    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False
        )

    print(f"✅ Generation done in {time.time() - start:.2f}s")

    # ---------------------------------------------
    # DECODE
    # ---------------------------------------------

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    # 🆕 NEW IN V8: Append current exchange to history
    chat_history.append({
        "user": question,
        "bot": answer
    })

    # ---------------------------------------------
    # OUTPUT
    # ---------------------------------------------

    print("\n==============================")
    print("📌 FINAL RESULT")
    print("==============================")

    print("\n📁 Sources:", ", ".join(sources))
    print("\n🤖 Bot:", answer)
    print()
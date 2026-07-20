import time
from core import RAGEngine

# =====================================================
# 🧠 CognIQ (Terminal Interface)
# =====================================================

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Configuration
QUANTIZATION_MODE = "4bit" 
USE_HYBRID = True

print(f"🔄 Starting CognIQ (Terminal Mode)...")

# Initialize Engine
engine = RAGEngine(model_name, embed_model_name, cross_encoder_model_name)

# Load Models
engine.load_models(quantization_mode=QUANTIZATION_MODE)

# Process Knowledge Base
engine.process_knowledge_base()

print("\n🤖 CognIQ Ready! Type 'quit' to exit.\n")

chat_history = []

while True:
    question = input("You: ").strip()
    if question.lower() == "quit":
        print("👋 Goodbye.")
        break
    if not question:
        continue

    print("\n🔍 Retrieving context...")
    top_chunks, metrics = engine.retrieve(question, k=3, use_hybrid=USE_HYBRID)

    if not top_chunks:
        print("❌ No relevant context found.\n")
        continue

    context = ""
    sources = set()
    for c in top_chunks:
        meta = c.get("metadata", {})
        meta_header = ""
        if meta:
            meta_fields = []
            if meta.get("author"): meta_fields.append(f"Author: {meta['author']}")
            if meta.get("created_time"): meta_fields.append(f"Created: {meta['created_time']}")
            if meta.get("modified_time"): meta_fields.append(f"Modified: {meta['modified_time']}")
            if meta_fields:
                meta_header = " | " + " | ".join(meta_fields)
        context += f"\n[Source: {c['source']}{meta_header} | Score: {c['score']:.3f}]\n{c['text'][:200]}...\n"
        sources.add(c["source"])

    print(f"✅ Retrieved from: {', '.join(sources)}")
    print(f"⏱️ Retrieval Latency: {sum(metrics.values()):.4f}s")

    print("\n🤖 Bot: ", end="", flush=True)
    streamer = engine.generate_stream(question, context, chat_history)
    
    full_response = ""
    start_time = time.time()
    for new_text in streamer:
        print(new_text, end="", flush=True)
        full_response += new_text
    
    print(f"\n\n✅ Generation done in {time.time() - start_time:.2f}s\n")

    chat_history.append({
        "user": question,
        "bot": full_response.strip()
    })
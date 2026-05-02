from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True
)

# Load all chunks from data folder
chunks = []

folder = "knowledge_source"

for filename in os.listdir(folder):
    path = os.path.join(folder, filename)

    if filename.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        for line in text.split("\n"):
            line = line.strip()
            if line:
                chunks.append({
                    "source": filename,
                    "text": line
                })

print("\nRAG Chatbot Ready!")
print("Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "quit":
        print("Goodbye.")
        break

    best_chunk = ""
    best_source = ""
    best_score = -1

    for item in chunks:
        chunk = item["text"]

        score = 0
        for word in question.lower().split():
            if word in chunk.lower():
                score += 1

        if score > best_score:
            best_score = score
            best_chunk = chunk
            best_source = item["source"]

    prompt = f"""
Answer only using the context below.

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
            max_new_tokens=40,
            do_sample=False
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    print(f"\nSource: {best_source}")
    print("Bot:", answer)
    print()
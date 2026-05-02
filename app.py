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

# Split into paragraph chunks
chunks = [c.strip() for c in text.split("\n") if c.strip()]

question = input("Ask question: ").lower()

# Score chunks by keyword overlap
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

print("\nBEST CHUNK:\n")
print(best_chunk)

print("\nRESULT:\n")
print(answer)
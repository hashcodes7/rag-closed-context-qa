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

# Read notes once
with open("notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Make chunks (1 line = 1 chunk)
chunks = [c.strip() for c in text.split("\n") if c.strip()]

print("\nRAG Chatbot Ready!")
print("Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "quit":
        print("Goodbye.")
        break

    # Retrieve best chunk
    best_chunk = ""
    best_score = -1

    for chunk in chunks:
        score = 0
        for word in question.lower().split():
            if word in chunk.lower():
                score += 1

        if score > best_score:
            best_score = score
            best_chunk = chunk

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

    # Show answer only after Answer:
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    print("\nBot:", answer)
    print()
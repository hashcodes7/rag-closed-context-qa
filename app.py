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

with open("notes.txt", "r", encoding="utf-8") as f:
    context = f.read()

question = input("Ask question: ")

prompt = f"""
Read the text below and answer only from it.
If answer is not present, say Not found.

Text:
{context}

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

print("\nRESULT:\n")
print(answer)
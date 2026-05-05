import os
import time
import math
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from sentence_transformers import SentenceTransformer, CrossEncoder
from threading import Thread

# =====================================================
# 🧠 RAG ENGINE CORE (v15 logic)
# =====================================================

class SimpleBM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            self.avgdl = 0
            self.doc_freqs = []
            self.idf = {}
            self.doc_len = []
            return
            
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        
        nd = {} 
        for doc in corpus:
            self.doc_len.append(len(doc))
            frequencies = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1
        
        for word, freq in nd.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query):
        if self.corpus_size == 0: return []
        scores = [0.0] * self.corpus_size
        for word in query:
            if word not in self.idf: continue
            idf = self.idf[word]
            for i in range(self.corpus_size):
                fi = self.doc_freqs[i].get(word, 0)
                scores[i] += idf * (fi * (self.k1 + 1)) / (fi + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
        return scores

def tokenize(text):
    return text.lower().replace(".", " ").replace(",", " ").replace("?", " ").split()

def recursive_chunk_text(text, chunk_size=1000, overlap=200):
    separators = ["\n\n", "\n", ". ", " "]
    def _split(text, separators):
        if not text.strip(): return []
        if len(text) <= chunk_size: return [text.strip()]
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
                if current.strip():
                    if len(current) > chunk_size and remaining_seps:
                        chunks.extend(_split(current, remaining_seps))
                    else:
                        chunks.append(current.strip())
                if chunks and overlap > 0:
                    last = chunks[-1]
                    overlap_text = last[-overlap:].strip()
                    current = (overlap_text + " " + part.strip()).strip()
                else:
                    current = part.strip()
        if current.strip():
            if len(current) > chunk_size and remaining_seps:
                chunks.extend(_split(current, remaining_seps))
            else:
                chunks.append(current.strip())
        return chunks
    return _split(text, separators)

def truncate(text, max_words=120):
    return " ".join(text.split()[:max_words])

def reciprocal_rank_fusion(results_list, k=60):
    fused_scores = {}
    for results in results_list:
        for rank, idx in enumerate(results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank)
    sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return sorted_indices

# --- CORE LOADER CLASS ---

class RAGEngine:
    def __init__(self, model_name, embed_model_name, cross_encoder_model_name):
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        
        self.tokenizer = None
        self.model = None
        self.embedder = None
        self.cross_encoder = None
        
        self.chunks = []
        self.faiss_index = None
        self.bm25_index = None

    def load_models(self, quantization_mode="4bit"):
        print(f"🔄 Loading models (Mode: {quantization_mode})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        quant_config = None
        if quantization_mode == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization_mode == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                low_cpu_mem_usage=True,
                device_map="auto" if quant_config else None
            )
        except Exception as e:
            print(f"⚠️ Quantization failed: {e}. Falling back to full precision.")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, low_cpu_mem_usage=True)
            
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)
        print("✅ Models loaded.")

    def process_knowledge_base(self, folder="knowledge_source", cache_file="vector_cache.pt", index_file="faiss_index.bin"):
        if os.path.exists(cache_file) and os.path.exists(index_file):
            print("💾 Loading cache...")
            self.chunks = torch.load(cache_file)["chunks"]
            self.faiss_index = faiss.read_index(index_file)
        else:
            print("📂 Processing files from scratch...")
            self.chunks = []
            for filename in os.listdir(folder):
                if not filename.endswith(".txt"): continue
                path = os.path.join(folder, filename)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                text_chunks = recursive_chunk_text(text)
                for i, chunk in enumerate(text_chunks):
                    self.chunks.append({"source": filename, "chunk_id": i, "text": chunk})
            
            chunk_texts = [item["text"] for item in self.chunks]
            embeddings_np = self.embedder.encode(chunk_texts, convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(embeddings_np)
            dim = embeddings_np.shape[1]
            self.faiss_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.faiss_index.hnsw.efConstruction = 200
            self.faiss_index.add(embeddings_np)
            
            torch.save({"chunks": self.chunks}, cache_file)
            faiss.write_index(self.faiss_index, index_file)
            print("✅ Knowledge base indexed.")

        # Always initialize BM25
        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25_index = SimpleBM25(tokenized_corpus)

    def retrieve(self, question, k=3, use_hybrid=True):
        metrics = {}
        
        # 1a. Semantic
        start = time.time()
        query_vec = self.embedder.encode([question], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        self.faiss_index.hnsw.efSearch = 64
        _, s_indices = self.faiss_index.search(query_vec, k=min(20, self.faiss_index.ntotal))
        semantic_ids = [int(idx) for idx in s_indices[0] if idx != -1]
        metrics["semantic_time"] = time.time() - start
        
        # 1b. Keyword
        start = time.time()
        tokenized_query = tokenize(question)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        keyword_ids = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
        metrics["keyword_time"] = time.time() - start
        
        # 1c. Fusion
        start = time.time()
        fused_ids = reciprocal_rank_fusion([semantic_ids, keyword_ids]) if use_hybrid else semantic_ids
        candidates = [self.chunks[idx].copy() for idx in fused_ids[:10]]
        metrics["fusion_time"] = time.time() - start
        
        # 2. Rerank
        start = time.time()
        cross_inp = [[question, item["text"]] for item in candidates]
        cross_scores = self.cross_encoder.predict(cross_inp)
        for i in range(len(candidates)):
            candidates[i]["score"] = float(cross_scores[i])
        candidates.sort(key=lambda x: x["score"], reverse=True)
        final_results = candidates[:k]
        metrics["rerank_time"] = time.time() - start
        
        return final_results, metrics

    def generate_stream(self, question, context, history, max_tokens=150):
        messages = [
            {"role": "system", "content": (
                "You are an assistant. Answer the user's question using ONLY the provided context.\n"
                f"<context>\n{context}\n</context>\n"
                "If the answer is not in the context, reply exactly with 'Not found.' Do not add explanations."
            )}
        ]
        for entry in history[-2:]:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["bot"]})
        messages.append({"role": "user", "content": question})
        
        text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_tokens, do_sample=False)
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        return streamer

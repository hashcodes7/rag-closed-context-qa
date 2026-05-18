import os
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import math
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from sentence_transformers import SentenceTransformer, CrossEncoder
from threading import Thread

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

# =====================================================
# 🧠 RAG ENGINE CORE (v19 Hybrid Logic)
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

import re
def tokenize(text):
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()

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

def semantic_chunk_text(text, embedder, threshold=0.5, max_chunk_size=1200):
    """
    Splits text into chunks based on semantic similarity between sentences.
    """
    import re
    # Treat newlines as sentence boundaries by replacing them with a period and space
    clean_text = re.sub(r'\n+', '. ', text)
    sentences = re.split(r'(?<=[.!?]) +', clean_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences: return []
    
    embeddings = embedder.encode(sentences, convert_to_numpy=True)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Calculate similarity with previous sentence
        sim = np.dot(embeddings[i], embeddings[i-1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]) + 1e-9)
        
        # Check if current chunk is getting too big or similarity is low
        current_text = " ".join(current_chunk)
        if sim < threshold or len(current_text) > max_chunk_size:
            chunks.append(current_text)
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def extract_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            import fitz
            text = ""
            doc = fitz.open(filepath)
            for page in doc:
                text += page.get_text() + "\n"
            return text
        elif ext == ".docx":
            import docx
            doc = docx.Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        elif ext in [".html", ".htm"]:
            from bs4 import BeautifulSoup
            with open(filepath, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ')
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                return '\n'.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        print(f"[!] Error reading {filepath}: {e}")
    return ""

def reciprocal_rank_fusion(results_list, k=60):
    fused_scores = {}
    for results in results_list:
        for rank, idx in enumerate(results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank)
    sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return sorted_indices

class RAGEngine:
    def __init__(self, model_name, embed_model_name, cross_encoder_model_name):
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        
        self.tokenizer = None
        self.model = None
        self.embedder = None
        self.cross_encoder = None
        self.is_gguf = False
        
        self.chunks = []
        self.parent_chunks = {} # Maps parent_id to text
        self.faiss_index = None
        self.bm25_index = None

    def load_models(self, quantization_mode="4bit"):
        has_cuda = torch.cuda.is_available()
        
        # Check for GGUF mode
        if "gguf" in self.model_name.lower() or self.model_name.endswith(".gguf"):
            if not HAS_LLAMA_CPP:
                raise ImportError("Please install llama-cpp-python to use GGUF models: pip install llama-cpp-python")
            
            print(f"[+] Loading GGUF Model: {self.model_name} (CPU Optimized)")
            if "/" in self.model_name and not os.path.exists(self.model_name):
                try:
                    from modelscope.hub.snapshot_download import snapshot_download
                    print(f"[*] Attempting download from ModelScope: {self.model_name}")
                    try:
                        model_dir = snapshot_download(self.model_name, allow_patterns=['*q4_k_m.gguf'])
                    except Exception as e:
                        print(f"[*] ModelScope failed with {self.model_name}: {e}")
                        if "/" in self.model_name:
                            parts = self.model_name.split("/")
                            parts[0] = parts[0].lower()
                            lowered_name = "/".join(parts)
                            print(f"[*] Trying lowered name on ModelScope: {lowered_name}")
                            model_dir = snapshot_download(lowered_name, allow_patterns=['*q4_k_m.gguf'])
                        else:
                            raise e
                    
                    gguf_file = None
                    for root, dirs, files in os.walk(model_dir):
                        for f in files:
                            if f.endswith('.gguf'):
                                gguf_file = os.path.join(root, f)
                                break
                        if gguf_file:
                            break
                                
                    if gguf_file:
                        print(f"[+] Loading ModelScope file: {gguf_file}")
                        self.model = Llama(model_path=gguf_file, n_ctx=2048, verbose=False)
                    else:
                        print("[!] No GGUF file found in ModelScope download. Falling back to Hugging Face.")
                        raise FileNotFoundError("No GGUF file found")
                except (ImportError, Exception) as e:
                    print(f"[!] ModelScope failed or not installed: {e}. Falling back to Hugging Face.")
                    self.model = Llama.from_pretrained(
                        repo_id=self.model_name,
                        filename="*q4_k_m.gguf", 
                        verbose=False,
                        n_ctx=2048,
                        n_threads=os.cpu_count() or 4
                    )
            else:
                self.model = Llama(model_path=self.model_name, n_ctx=2048, verbose=False)
            self.is_gguf = True
            self.tokenizer = None # Llama handles tokenization
        else:
            print(f"[*] Loading Transformers Model (Device: {'GPU' if has_cuda else 'CPU'})")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.is_gguf = False
            
            quant_config = None
            if has_cuda:
                if quantization_mode == "4bit":
                    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
                elif quantization_mode == "8bit":
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    low_cpu_mem_usage=True,
                    device_map="auto" if (has_cuda and quant_config) else None,
                    torch_dtype="auto"
                )
            except Exception as e:
                print(f"[!] Model load failed: {e}. Falling back to defaults.")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, low_cpu_mem_usage=True)
            
        # Load Embedder
        local_embed_path = os.path.join(os.getcwd(), "all-MiniLM-L6-v2")
        cached_embed_path = os.path.expanduser("~/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2")
        std_cached_embed_path = os.path.expanduser("~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")
        
        target_embed_path = None
        if os.path.exists(local_embed_path):
            target_embed_path = local_embed_path
        elif os.path.exists(cached_embed_path) and (os.path.exists(os.path.join(cached_embed_path, "model.safetensors")) or os.path.exists(os.path.join(cached_embed_path, "pytorch_model.bin"))):
            target_embed_path = cached_embed_path
        elif os.path.exists(std_cached_embed_path):
            target_embed_path = std_cached_embed_path

        if target_embed_path:
            print(f"[+] Loading embedder from: {target_embed_path}")
            self.embedder = SentenceTransformer(target_embed_path)
        else:
            try:
                from modelscope.hub.snapshot_download import snapshot_download
                print(f"[*] Downloading embedder from ModelScope: {self.embed_model_name}")
                try:
                    embed_dir = snapshot_download(self.embed_model_name)
                except Exception:
                    if "/" in self.embed_model_name:
                        parts = self.embed_model_name.split("/")
                        parts[0] = parts[0].lower()
                        lowered_name = "/".join(parts)
                        print(f"[*] Trying lowered name on ModelScope: {lowered_name}")
                        embed_dir = snapshot_download(lowered_name)
                    else:
                        raise
                self.embedder = SentenceTransformer(embed_dir)
            except Exception as e:
                print(f"[!] ModelScope embedder download failed: {e}. Falling back to Hugging Face.")
                self.embedder = SentenceTransformer(self.embed_model_name)

        # Load Cross-Encoder (Disabled due to network issues)
        print("[!] Disabling Cross-Encoder loading due to persistent connection issues.")
        self.cross_encoder = None
        print("[+] Models loaded.")

    def process_knowledge_base(self, folder="knowledge_source", cache_file="vector_cache.pt", index_file="faiss_index.bin", force_reindex=False, chunking_mode="semantic"):
        if not force_reindex and os.path.exists(cache_file) and os.path.exists(index_file):
            print("[*] Loading cache...")
            data = torch.load(cache_file)
            self.chunks = data["chunks"]
            self.parent_chunks = data.get("parent_chunks", {})
            self.faiss_index = faiss.read_index(index_file)
        else:
            print(f"[*] Processing files using {chunking_mode} chunking...")
            if force_reindex:
                if os.path.exists(cache_file): os.remove(cache_file)
                if os.path.exists(index_file): os.remove(index_file)
            
            self.chunks = []
            self.parent_chunks = {}
            if not os.path.exists(folder): os.makedirs(folder)
            
            valid_extensions = (".txt", ".pdf", ".docx", ".html", ".htm")
            parent_id_counter = 0
            
            for filename in os.listdir(folder):
                if not filename.lower().endswith(valid_extensions): continue
                path = os.path.join(folder, filename)
                text = extract_text_from_file(path)
                if not text.strip(): continue
                
                # 1. Create Semantic Parents
                if chunking_mode == "semantic":
                    parents = semantic_chunk_text(text, self.embedder)
                else:
                    parents = recursive_chunk_text(text, chunk_size=1500)
                
                for p_text in parents:
                    p_id = f"p_{parent_id_counter}"
                    self.parent_chunks[p_id] = p_text
                    parent_id_counter += 1
                    
                    # 2. Create Overlapping Children for dense retrieval
                    children = recursive_chunk_text(p_text, chunk_size=400, overlap=50)
                    for i, c_text in enumerate(children):
                        self.chunks.append({
                            "source": filename, 
                            "chunk_id": i, 
                            "text": c_text, 
                            "parent_id": p_id
                        })
            
            chunk_texts = [item["text"] for item in self.chunks]
            embeddings_np = self.embedder.encode(chunk_texts, convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(embeddings_np)
            dim = embeddings_np.shape[1]
            self.faiss_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.faiss_index.hnsw.efConstruction = 200
            self.faiss_index.add(embeddings_np)
            torch.save({"chunks": self.chunks, "parent_chunks": self.parent_chunks}, cache_file)
            faiss.write_index(self.faiss_index, index_file)
            print("[+] Knowledge base indexed.")

        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25_index = SimpleBM25(tokenized_corpus)

    def generate_hypothetical_answer(self, question):
        """Generates a brief hypothetical answer to improve retrieval."""
        prompt = f"Write a one-sentence technical answer to this question: {question}"
        
        if self.is_gguf:
            res = self.model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50
            )
            return res['choices'][0]['message']['content']
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def retrieve(self, question, k=3, use_hybrid=True, use_hyde=False, use_rerank=True, use_parent=True):
        metrics = {}
        start = time.time()
        
        # 1. Semantic Search (with optional HyDE)
        search_query = question
        if use_hyde:
            try:
                hyde_answer = self.generate_hypothetical_answer(question)
                search_query = f"{question} {hyde_answer}"
                metrics["hyde_gen_time"] = time.time() - start
            except Exception as e:
                print(f"[!] HyDE failed: {e}")
        
        query_vec = self.embedder.encode([search_query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        self.faiss_index.hnsw.efSearch = 64
        _, s_indices = self.faiss_index.search(query_vec, k=min(20, self.faiss_index.ntotal))
        semantic_ids = [int(idx) for idx in s_indices[0] if idx != -1]
        metrics["semantic_time"] = time.time() - start
        
        # 2. Keyword Search (BM25)
        start = time.time()
        if use_hybrid:
            tokenized_query = tokenize(question)
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            keyword_ids = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
        else:
            keyword_ids = []
        metrics["keyword_time"] = time.time() - start
        
        # 3. Hybrid Fusion (RRF)
        start = time.time()
        fused_ids = reciprocal_rank_fusion([semantic_ids, keyword_ids]) if use_hybrid else semantic_ids
        candidates = [self.chunks[idx].copy() for idx in fused_ids[:20]] # Keep more for reranking
        metrics["fusion_time"] = time.time() - start
        
        # 4. Reranking (Cross-Encoder)
        start = time.time()
        if use_rerank and self.cross_encoder is not None and len(candidates) > 0:
            cross_inp = [[question, item["text"]] for item in candidates]
            cross_scores = self.cross_encoder.predict(cross_inp)
            for i in range(len(candidates)):
                candidates[i]["score"] = float(cross_scores[i])
            candidates.sort(key=lambda x: x["score"], reverse=True)
        else:
            # If no reranking, scores are just their rank position
            for i, c in enumerate(candidates):
                c["score"] = 1.0 / (i + 1)
        
        final_results = candidates[:k]
        
        # 5. Parent-Document Expansion
        for item in final_results:
            if use_parent:
                p_id = item.get("parent_id")
                if p_id in self.parent_chunks:
                    item["retrieval_text"] = item["text"]
                    item["text"] = self.parent_chunks[p_id]
            else:
                item["retrieval_text"] = item["text"]
        
        metrics["rerank_time"] = time.time() - start
        return final_results, metrics

    def generate_stream(self, question, context, history, max_tokens=150):
        system_msg = (
            "You are a strict question-answering AI. You must answer the user's question based ONLY on the following Context.\n\n"
            f"--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
            "If the Context contains the answer, extract it and cite the source like [Source 1]. "
            "If the Context does NOT contain the answer, you must reply EXACTLY with 'Not found.' and nothing else."
        )
        
        messages = [{"role": "system", "content": system_msg}]
        for entry in history[-2:]:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["bot"]})
        messages.append({"role": "user", "content": question})
        
        if self.is_gguf:
            # GGUF Streamer (Generator)
            def _gguf_generator():
                for chunk in self.model.create_chat_completion(messages=messages, stream=True, max_tokens=max_tokens):
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        yield delta['content']
            return _gguf_generator()
        else:
            # Transformers Streamer
            text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.model.device)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_tokens, do_sample=False)
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            return streamer

import os
import time
import math
import torch
import faiss
import numpy as np
import fitz  # PyMuPDF
import docx  # python-docx
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

def semantic_chunk_text(text, embedder, threshold=0.5, max_chunk_size=1200):
    """
    Splits text into chunks based on semantic similarity between sentences.
    """
    import re
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' '))
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
            text = ""
            doc = fitz.open(filepath)
            for page in doc:
                text += page.get_text() + "\n"
            return text
        elif ext == ".docx":
            doc = docx.Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".html":
            try:
                from bs4 import BeautifulSoup
                with open(filepath, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    # Extract text, separate block elements with newlines
                    return soup.get_text(separator="\n", strip=True)
            except ImportError:
                print("[!] BeautifulSoup4 is required for HTML parsing. Run `pip install beautifulsoup4`")
                return ""
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
        
        # Check for Gemini API mode
        if self.model_name.startswith("gemini-"):
            print(f"[+] Using Google Gemini API: {self.model_name}")
            self.model = "api"
            self.is_gguf = False
            self.tokenizer = None
        # Check for GGUF mode
        elif "gguf" in self.model_name.lower() or self.model_name.endswith(".gguf"):
            if not HAS_LLAMA_CPP:
                raise ImportError("Please install llama-cpp-python to use GGUF models: pip install llama-cpp-python")
            
            print(f"[+] Loading GGUF Model: {self.model_name} (CPU Optimized)")
            
            import multiprocessing
            
            local_files = []
            if os.path.exists("models"):
                target_prefix = self.model_name.replace('/', '_').lower()
                for existing_file in os.listdir("models"):
                    if existing_file.lower().startswith(target_prefix) and "q4_k_m.gguf" in existing_file.lower():
                        local_files.append(os.path.join("models", existing_file))
                        break
            
            # Use physical cores to prevent thread contention
            optimal_threads = max(1, multiprocessing.cpu_count() // 2)
            
            if local_files:
                print(f"[+] Found local model file: {local_files[0]}")
                self.model = Llama(model_path=local_files[0], n_ctx=2048, n_threads=optimal_threads, verbose=False)
            elif "/" in self.model_name and not os.path.exists(self.model_name):
                 self.model = Llama.from_pretrained(
                    repo_id=self.model_name,
                    filename="*q4_k_m.gguf", 
                    verbose=False,
                    n_ctx=2048,
                    n_threads=optimal_threads
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
                import traceback
                print(f"[!] Model load failed:\n{traceback.format_exc()}")
                print("Falling back to defaults.")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, low_cpu_mem_usage=True)
            
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)
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
            
            # Accept common document formats including HTML/HTM
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

            # If no chunks were produced (e.g., unsupported files only), bail out gracefully
            if len(chunk_texts) == 0:
                print("[!] No text chunks were extracted from knowledge sources. Skipping index creation.")
                self.faiss_index = None
                # Create an empty BM25 index to avoid None checks elsewhere
                self.bm25_index = SimpleBM25([])
                return

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
                import traceback
                print(f"[!] HyDE failed:\n{traceback.format_exc()}")
        
        # If FAISS index isn't available (e.g., no KB files indexed), skip semantic search
        if self.faiss_index is not None and getattr(self.faiss_index, "ntotal", 0) > 0:
            query_vec = self.embedder.encode([search_query], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(query_vec)
            # Some FAISS index types may not expose .hnsw; guard defensively
            try:
                self.faiss_index.hnsw.efSearch = 64
            except Exception:
                pass
            _, s_indices = self.faiss_index.search(query_vec, k=min(20, self.faiss_index.ntotal))
            semantic_ids = [int(idx) for idx in s_indices[0] if idx != -1]
            metrics["semantic_time"] = time.time() - start
        else:
            semantic_ids = []
            metrics["semantic_time"] = 0.0
        
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
        if use_rerank and len(candidates) > 0:
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

    def generate_stream(self, question, context, history, max_tokens=150, api_key=None):
        system_msg = (
            "You are a corporate chatbot for Fresenius Medical Care made by Harsh Verma from Cognizant Technology Solutions. Answer the user's question using ONLY the provided context.\n"
            "Respond in a professional, concise, and corporate tone appropriate for an internal Fresenius Medical Care assistant.\n"
            "CRITICAL: Use in-text citations like [1], [2] to indicate which part of the context your answer came from when you reference it.\n"
            f"<context>\n{context}\n</context>\n"
            "If the answer is not contained in the provided context, reply exactly with \"I think this info isnt yet added to my knowledge base.\" Do not add explanations, speculation, or additional content."
        )
        
        if self.model_name.startswith("gemini-"):
            import google.generativeai as genai
            if not api_key:
                yield "Error: Google API Key is required for Gemini models."
                return
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model_name)
            
            prompt = system_msg + "\n\n"
            for entry in history[-2:]:
                prompt += f"User: {entry['user']}\nAssistant: {entry['bot']}\n"
            prompt += f"User: {question}\nAssistant:"
            
            try:
                response = model.generate_content(prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                import traceback
                print(f"[!] API Error:\n{traceback.format_exc()}")
                yield f"\n[API Error: {str(e)}]"
            return

        messages = [{"role": "system", "content": system_msg}]
        for entry in history[-2:]:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["bot"]})
        messages.append({"role": "user", "content": question})
        
        if self.is_gguf:
            # GGUF Streamer (Generator)
            for chunk in self.model.create_chat_completion(messages=messages, stream=True, max_tokens=max_tokens):
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    yield delta['content']
            return
        else:
            # Transformers Streamer
            text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.model.device)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_tokens, do_sample=False)
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            for text in streamer:
                yield text
            return

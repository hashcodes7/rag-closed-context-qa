import os
import time
import math
from datetime import datetime
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
    # Split on newlines, tab characters, and standard sentence boundaries
    sentences = re.split(r'\n+|(?<=[.!?]) +', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences: return []
    
    # Ensure no individual sentence exceeds max_chunk_size
    split_sentences = []
    for s in sentences:
        if len(s) > max_chunk_size:
            split_sentences.extend(recursive_chunk_text(s, chunk_size=max_chunk_size, overlap=200))
        else:
            split_sentences.append(s)
    sentences = split_sentences
    
    embeddings = embedder.encode(sentences, convert_to_numpy=True)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Calculate similarity with previous sentence
        sim = np.dot(embeddings[i], embeddings[i-1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]) + 1e-9)
        
        # Check if current chunk is getting too big or similarity is low
        current_text = " ".join(current_chunk)
        if sim < threshold or (len(current_text) + len(sentences[i]) + 1) > max_chunk_size:
            chunks.append(current_text)
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def extract_text_from_excel(filepath):
    try:
        import openpyxl
    except ImportError:
        print("[!] openpyxl is required for Excel parsing. Run `pip install openpyxl`")
        return ""
        
    wb = None
    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        text_parts = []
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            
            # Find the header row (first non-empty row)
            header_row = None
            header_idx = 0
            for idx, r in enumerate(rows):
                if any(cell is not None for cell in r):
                    header_row = r
                    header_idx = idx
                    break
            
            if header_row is None:
                continue
            
            headers = []
            for col_idx, cell in enumerate(header_row):
                if cell is not None and str(cell).strip():
                    headers.append(str(cell).strip())
                else:
                    headers.append(f"Column_{openpyxl.utils.get_column_letter(col_idx + 1)}")
            
            sheet_text = []
            for row_idx, r in enumerate(rows[header_idx + 1:], start=header_idx + 2):
                if not any(cell is not None for cell in r):
                    continue # Skip empty rows
                
                row_parts = []
                for col_idx, cell in enumerate(r):
                    if col_idx < len(headers):
                        val = str(cell).strip() if cell is not None else ""
                        if val:
                            row_parts.append(f"{headers[col_idx]}: {val}")
                
                if row_parts:
                    sheet_text.append(f"Sheet: {sheet_name} | Row {row_idx}: " + " | ".join(row_parts))
            
            if sheet_text:
                text_parts.append(f"--- Sheet: {sheet_name} ---\n" + "\n".join(sheet_text))
                
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"[!] Error reading Excel file {filepath}: {e}")
        return ""
    finally:
        if wb is not None:
            wb.close()

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
        elif ext in (".xlsx", ".xlsm"):
            return extract_text_from_excel(filepath)
        elif ext == ".html":
            try:
                from bs4 import BeautifulSoup
                with open(filepath, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    return soup.get_text(separator="\n", strip=True)
            except ImportError:
                print("[!] BeautifulSoup4 is required for HTML parsing. Run `pip install beautifulsoup4`")
                return ""
    except Exception as e:
        print(f"[!] Error reading {filepath}: {e}")
    return ""

def extract_metadata_and_text(filepath):
    import os
    from datetime import datetime
    ext = os.path.splitext(filepath)[1].lower()
    
    # 1. Base system metadata
    meta = {
        "filename": os.path.basename(filepath),
        "filepath": filepath.replace('\\', '/'),
        "file_size_bytes": os.path.getsize(filepath),
        "created_time": datetime.fromtimestamp(os.path.getctime(filepath)).strftime('%Y-%m-%d %H:%M:%S'),
        "modified_time": datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S'),
        "author": "",
        "creator": "",
        "title": "",
        "subject": "",
        "keywords": "",
        "description": ""
    }
    
    text = ""
    try:
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == ".pdf":
            doc = fitz.open(filepath)
            for page in doc:
                text += page.get_text() + "\n"
            
            # Extract PDF metadata
            pdf_meta = doc.metadata
            if pdf_meta:
                meta["author"] = pdf_meta.get("author") or ""
                meta["creator"] = pdf_meta.get("creator") or ""
                meta["title"] = pdf_meta.get("title") or ""
                meta["subject"] = pdf_meta.get("subject") or ""
                meta["keywords"] = pdf_meta.get("keywords") or ""
                
                # Format internal PDF creation dates if available (typically D:YYYYMMDDHHMMSS)
                c_date = pdf_meta.get("creationDate")
                if c_date and c_date.startswith("D:"):
                    try:
                        date_str = c_date[2:16]
                        dt = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                        meta["created_time"] = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        pass
                m_date = pdf_meta.get("modDate")
                if m_date and m_date.startswith("D:"):
                    try:
                        date_str = m_date[2:16]
                        dt = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                        meta["modified_time"] = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        pass
            doc.close()
            
        elif ext == ".docx":
            doc = docx.Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
            
            # Extract DOCX core properties
            props = doc.core_properties
            if props:
                meta["author"] = props.author or ""
                meta["creator"] = props.last_modified_by or ""
                meta["title"] = props.title or ""
                meta["subject"] = props.subject or ""
                meta["keywords"] = props.keywords or ""
                
                if props.created:
                    try:
                        meta["created_time"] = props.created.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        pass
                if props.modified:
                    try:
                        meta["modified_time"] = props.modified.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        pass
                        
        elif ext in (".xlsx", ".xlsm"):
            text = extract_text_from_excel(filepath)
            
            # Extract Excel properties
            try:
                import openpyxl
                wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
                props = wb.properties
                if props:
                    meta["author"] = props.creator or ""
                    meta["creator"] = props.lastModifiedBy or ""
                    meta["title"] = props.title or ""
                    meta["subject"] = props.subject or ""
                    meta["keywords"] = props.keywords or ""
                    meta["description"] = props.description or ""
                    
                    if props.created:
                        meta["created_time"] = props.created.strftime('%Y-%m-%d %H:%M:%S')
                    if props.modified:
                        meta["modified_time"] = props.modified.strftime('%Y-%m-%d %H:%M:%S')
                wb.close()
            except Exception as e:
                print(f"[!] Error reading Excel metadata for {filepath}: {e}")
                
        elif ext in (".html", ".htm"):
            try:
                from bs4 import BeautifulSoup
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    html_content = f.read()
                    soup = BeautifulSoup(html_content, "html.parser")
                    text = soup.get_text(separator="\n", strip=True)
                    
                    # Extract title
                    if soup.title and soup.title.string:
                        meta["title"] = soup.title.string.strip()
                        
                    # Extract meta tags
                    meta_author = soup.find("meta", attrs={"name": "author"})
                    if meta_author:
                        meta["author"] = meta_author.get("content", "").strip()
                        
                    meta_desc = soup.find("meta", attrs={"name": "description"})
                    if meta_desc:
                        meta["description"] = meta_desc.get("content", "").strip()
                        
                    meta_keywords = soup.find("meta", attrs={"name": "keywords"})
                    if meta_keywords:
                        meta["keywords"] = meta_keywords.get("content", "").strip()
            except ImportError:
                print("[!] BeautifulSoup4 is required for HTML parsing. Run `pip install beautifulsoup4`")
    except Exception as e:
        print(f"[!] Error reading {filepath}: {e}")
        
    return text, meta

def format_metadata_header(meta):
    header = "[Document Metadata]\n"
    header += f"- Filename: {meta['filename']}\n"
    header += f"- Path: {meta['filepath']}\n"
    header += f"- Size: {meta['file_size_bytes'] / 1024:.1f} KB\n"
    header += f"- Created: {meta['created_time']}\n"
    header += f"- Modified: {meta['modified_time']}\n"
    if meta["author"]:
        header += f"- Author: {meta['author']}\n"
    if meta["creator"]:
        header += f"- Creator/Editor: {meta['creator']}\n"
    if meta["title"]:
        header += f"- Title: {meta['title']}\n"
    if meta["subject"]:
        header += f"- Subject: {meta['subject']}\n"
    if meta["keywords"]:
        header += f"- Keywords: {meta['keywords']}\n"
    if meta["description"]:
        header += f"- Description: {meta['description']}\n"
    header += "------------------\n\n"
    return header

def reciprocal_rank_fusion(results_list, k=60):
    fused_scores = {}
    for results in results_list:
        for rank, idx in enumerate(results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank)
def is_small_talk(query):
    """Detect if a user query is a simple greeting, salutation, or casual small talk."""
    import re
    q = query.strip().lower()
    clean = re.sub(r'[^\w\s]', '', q).strip()
    words = clean.split()
    
    greetings = {
        "hi", "hello", "hey", "heya", "greetings", "good morning", "good afternoon",
        "good evening", "howdy", "hola", "sup", "whats up", "whatsup",
        "how are you", "how are you doing", "hows it going", "how is it going",
        "who are you", "what can you do", "what are you", "help", "thanks",
        "thank you", "thankyou", "bye", "goodbye", "good day", "namaste"
    }
    
    if len(words) <= 5:
        if clean in greetings:
            return True
        for g in greetings:
            if clean.startswith(g) or clean.endswith(g):
                return True
    return False

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
        self.n_ctx = 8192
        
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
                self.model = Llama(model_path=local_files[0], n_ctx=self.n_ctx, n_threads=optimal_threads, verbose=False)
            elif "/" in self.model_name and not os.path.exists(self.model_name):
                 self.model = Llama.from_pretrained(
                    repo_id=self.model_name,
                    filename="*q4_k_m.gguf", 
                    verbose=False,
                    n_ctx=self.n_ctx,
                    n_threads=optimal_threads
                )
            else:
                self.model = Llama(model_path=self.model_name, n_ctx=self.n_ctx, verbose=False)
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

    def process_knowledge_base(self, folder="knowledge_source", cache_file="vector_cache.pt", index_file="faiss_index.bin", force_reindex=False, chunking_mode="semantic", incremental=True, progress_callback=None):
        import json
        registry_file = "file_registry.json"
        valid_extensions = (".txt", ".pdf", ".docx", ".html", ".htm", ".xlsx", ".xlsm")

        def notify_progress(pct, text):
            if progress_callback:
                try:
                    progress_callback(min(1.0, max(0.0, float(pct))), text)
                except Exception:
                    pass

        notify_progress(0.02, "Initializing indexing process...")

        # helper function to load registry
        def load_registry():
            if os.path.exists(registry_file):
                try:
                    with open(registry_file, "r") as f:
                        return json.load(f)
                except Exception:
                    pass
            return {}

        # helper function to save registry
        def save_registry(registry_dict):
            try:
                with open(registry_file, "w") as f:
                    json.dump(registry_dict, f, indent=4)
            except Exception as e:
                print(f"[!] Error saving registry: {e}")

        # Determine if we should perform an incremental update or a full reindex
        do_incremental = incremental and not force_reindex and os.path.exists(cache_file) and os.path.exists(index_file)

        if do_incremental:
            print("[*] Running Incremental Update...")
            notify_progress(0.05, "Loading existing vector cache and index...")
            try:
                data = torch.load(cache_file, weights_only=False)
                self.chunks = data.get("chunks", [])
                self.parent_chunks = data.get("parent_chunks", {})
                self.faiss_index = faiss.read_index(index_file)
                
                # Dynamic upgrade: if old cache does not have embeddings_np, generate it
                embeddings_np = data.get("embeddings_np", None)
                if embeddings_np is None and len(self.chunks) > 0:
                    print("[*] Upgrading old cache file: generating embeddings for existing chunks...")
                    notify_progress(0.08, "Upgrading cache: generating embeddings...")
                    chunk_texts = [item["text"] for item in self.chunks]
                    embeddings_np = self.embedder.encode(chunk_texts, convert_to_numpy=True).astype("float32")
                    faiss.normalize_L2(embeddings_np)
                
                # Upgrade cache: populate missing metadata field
                updated_cache = False
                for c in self.chunks:
                    if "metadata" not in c:
                        full_path = os.path.join(folder, c["source"])
                        if os.path.exists(full_path):
                            try:
                                _, meta = extract_metadata_and_text(full_path)
                                c["metadata"] = meta
                                updated_cache = True
                            except Exception:
                                c["metadata"] = {}
                        else:
                            c["metadata"] = {}
                
                if updated_cache or data.get("embeddings_np", None) is None:
                    print("[*] Saving upgraded cache file...")
                    torch.save({
                        "chunks": self.chunks,
                        "parent_chunks": self.parent_chunks,
                        "embeddings_np": embeddings_np
                    }, cache_file)
            except Exception as e:
                print(f"[!] Failed to load existing index/cache ({e}). Falling back to complete re-index.")
                do_incremental = False

        if do_incremental:
            notify_progress(0.12, "Scanning knowledge base for modified or new files...")
            # 1. Load registry or bootstrap it
            registry = load_registry()
            
            # Bootstrapping fault tolerance: if registry is empty but cache has data, build registry from cache
            if not registry and len(self.chunks) > 0:
                print("[*] Bootstrapping registry from existing vector cache...")
                # Map relative paths of currently cached files to current mtime on disk (if they exist)
                for c in self.chunks:
                    source_path = c["source"]
                    full_path = os.path.join(folder, source_path)
                    if os.path.exists(full_path) and source_path not in registry:
                        registry[source_path] = os.path.getmtime(full_path)
                save_registry(registry)

            # 2. Check disk for changed/new/deleted files
            current_files = {}
            for root, dirs, files in os.walk(folder):
                for filename in files:
                    if filename.lower().endswith(valid_extensions):
                        full_path = os.path.join(root, filename)
                        relpath = os.path.relpath(full_path, folder).replace('\\', '/')
                        current_files[relpath] = os.path.getmtime(full_path)

            new_or_modified = []
            deleted_files = []

            # Find new or modified files
            for relpath, mtime in current_files.items():
                if relpath not in registry or mtime > registry.get(relpath, 0):
                    new_or_modified.append(relpath)

            # Find deleted files (in registry but no longer on disk)
            for relpath in list(registry.keys()):
                if relpath not in current_files:
                    deleted_files.append(relpath)

            if not new_or_modified and not deleted_files:
                print("[+] Knowledge base is up to date. No files changed.")
                notify_progress(0.90, "Building BM25 keyword index...")
                tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
                self.bm25_index = SimpleBM25(tokenized_corpus)
                notify_progress(1.0, "Knowledge base is already up to date!")
                return

            print(f"[*] Incremental status: {len(new_or_modified)} new/modified, {len(deleted_files)} deleted.")

            # 3. Handle deletions/modifications in chunks, parent_chunks, and embeddings
            # We keep only chunks that are NOT from deleted or modified files
            keep_mask = []
            filtered_chunks = []
            for idx, c in enumerate(self.chunks):
                if c["source"] not in deleted_files and c["source"] not in new_or_modified:
                    keep_mask.append(idx)
                    filtered_chunks.append(c)

            # Filter parent chunks
            keep_parent_ids = {c["parent_id"] for c in filtered_chunks}
            self.parent_chunks = {k: v for k, v in self.parent_chunks.items() if k in keep_parent_ids}

            # Filter embeddings array
            if embeddings_np is not None and len(keep_mask) > 0:
                remaining_embeddings = embeddings_np[keep_mask]
            else:
                remaining_embeddings = np.empty((0, self.embedder.get_sentence_embedding_dimension()), dtype="float32")

            self.chunks = filtered_chunks

            # Remove deleted files from registry
            for df in deleted_files:
                if df in registry:
                    del registry[df]

            # 4. Process new/modified files
            new_chunks_added = []
            # Find the highest counter for parent_id to avoid collisions
            parent_id_counter = 0
            for p_id in self.parent_chunks.keys():
                try:
                    num = int(p_id.split('_')[1])
                    if num >= parent_id_counter:
                        parent_id_counter = num + 1
                except Exception:
                    pass

            total_inc_files = len(new_or_modified)
            for file_idx, relpath in enumerate(new_or_modified):
                pct = 0.15 + ((file_idx + 1) / max(1, total_inc_files)) * 0.50
                notify_progress(pct, f"Processing file ({file_idx+1}/{total_inc_files}): {relpath}")

                full_path = os.path.join(folder, relpath)
                if not os.path.exists(full_path):
                    continue

                raw_text, meta = extract_metadata_and_text(full_path)
                if not raw_text.strip():
                    print(f"[!] Skipping {relpath}: no text extracted")
                    continue

                print(f"[+] Processing {relpath}...")
                parts = relpath.split('/')
                namespace = parts[0] if len(parts) > 1 else (parts[0] if parts else 'root')

                metadata_header = format_metadata_header(meta)
                text = metadata_header + raw_text

                if chunking_mode == "semantic":
                    parents = semantic_chunk_text(text, self.embedder)
                else:
                    parents = recursive_chunk_text(text, chunk_size=1500)

                for p_text in parents:
                    p_id = f"p_{parent_id_counter}"
                    self.parent_chunks[p_id] = p_text
                    parent_id_counter += 1

                    children = recursive_chunk_text(p_text, chunk_size=400, overlap=50)
                    for i, c_text in enumerate(children):
                        new_chunks_added.append({
                            "source": relpath,
                            "chunk_id": i,
                            "text": c_text,
                            "parent_id": p_id,
                            "namespace": namespace,
                            "metadata": meta
                        })

                # Update registry mtime
                registry[relpath] = current_files[relpath]

            # 5. Embed new chunks and merge with remaining ones
            if new_chunks_added:
                notify_progress(0.70, f"Generating vector embeddings for {len(new_chunks_added)} new chunks...")
                new_texts = [item["text"] for item in new_chunks_added]
                new_embeds = self.embedder.encode(new_texts, convert_to_numpy=True).astype("float32")
                faiss.normalize_L2(new_embeds)
                
                if remaining_embeddings.shape[0] > 0:
                    embeddings_np = np.concatenate([remaining_embeddings, new_embeds], axis=0)
                else:
                    embeddings_np = new_embeds

                self.chunks.extend(new_chunks_added)
            else:
                embeddings_np = remaining_embeddings

            # Ensure namespace exists on all chunks
            for c in self.chunks:
                if "namespace" not in c:
                    c["namespace"] = "root"

            # 6. Rebuild and save FAISS and cache
            if embeddings_np.shape[0] > 0:
                notify_progress(0.85, "Rebuilding FAISS HNSW index...")
                dim = embeddings_np.shape[1]
                self.faiss_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
                self.faiss_index.hnsw.efConstruction = 200
                self.faiss_index.add(embeddings_np)
                faiss.write_index(self.faiss_index, index_file)
                notify_progress(0.92, "Saving vector cache to disk...")
                torch.save({
                    "chunks": self.chunks, 
                    "parent_chunks": self.parent_chunks,
                    "embeddings_np": embeddings_np
                }, cache_file)
                save_registry(registry)
                print(f"[+] Knowledge base updated. Total chunks: {len(self.chunks)}.")
            else:
                print("[!] Knowledge base is empty. Clearing index files.")
                self.faiss_index = None
                if os.path.exists(cache_file): os.remove(cache_file)
                if os.path.exists(index_file): os.remove(index_file)
                if os.path.exists(registry_file): os.remove(registry_file)

        else:
            # Complete Reindex Flow
            print(f"[*] Processing files using {chunking_mode} chunking (Complete Re-Index)...")
            notify_progress(0.05, "Preparing full re-index environment...")
            
            # Wipe files
            if os.path.exists(cache_file): os.remove(cache_file)
            if os.path.exists(index_file): os.remove(index_file)
            if os.path.exists(registry_file): os.remove(registry_file)
            
            self.chunks = []
            self.parent_chunks = {}
            if not os.path.exists(folder): os.makedirs(folder)
            
            parent_id_counter = 0
            registry = {}

            all_files = []
            for root, dirs, files in os.walk(folder):
                for filename in files:
                    if filename.lower().endswith(valid_extensions):
                        all_files.append(os.path.join(root, filename))

            total_files = len(all_files)
            for file_idx, path in enumerate(all_files):
                relpath = os.path.relpath(path, folder).replace('\\', '/')
                pct = 0.05 + ((file_idx + 1) / max(1, total_files)) * 0.55
                notify_progress(pct, f"Extracting & chunking ({file_idx+1}/{total_files}): {relpath}")

                parts = relpath.split('/')
                namespace = parts[0] if len(parts) > 1 else (parts[0] if parts else 'root')

                raw_text, meta = extract_metadata_and_text(path)
                if not raw_text.strip():
                    print(f"[!] Skipping {relpath}: no text extracted")
                    continue

                print(f"[+] Processing {relpath} (namespace={namespace})...")
                metadata_header = format_metadata_header(meta)
                text = metadata_header + raw_text
                
                if chunking_mode == "semantic":
                    parents = semantic_chunk_text(text, self.embedder)
                else:
                    parents = recursive_chunk_text(text, chunk_size=1500)

                for p_text in parents:
                    p_id = f"p_{parent_id_counter}"
                    self.parent_chunks[p_id] = p_text
                    parent_id_counter += 1

                    children = recursive_chunk_text(p_text, chunk_size=400, overlap=50)
                    for i, c_text in enumerate(children):
                        self.chunks.append({
                            "source": relpath,
                            "chunk_id": i,
                            "text": c_text,
                            "parent_id": p_id,
                            "namespace": namespace,
                            "metadata": meta
                        })

                registry[relpath] = os.path.getmtime(path)

            chunk_texts = [item["text"] for item in self.chunks]

            if len(chunk_texts) == 0:
                print("[!] No text chunks were extracted from knowledge sources. Skipping index creation.")
                self.faiss_index = None
                self.bm25_index = SimpleBM25([])
                notify_progress(1.0, "Knowledge base is empty.")
                return

            notify_progress(0.65, f"Generating vector embeddings for {len(chunk_texts)} chunks...")
            embeddings_np = self.embedder.encode(chunk_texts, convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(embeddings_np)
            dim = embeddings_np.shape[1]
            
            notify_progress(0.85, "Constructing FAISS HNSW index...")
            self.faiss_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.faiss_index.hnsw.efConstruction = 200
            self.faiss_index.add(embeddings_np)
            
            notify_progress(0.92, "Saving vector cache & file registry...")
            torch.save({
                "chunks": self.chunks, 
                "parent_chunks": self.parent_chunks,
                "embeddings_np": embeddings_np
            }, cache_file)
            faiss.write_index(self.faiss_index, index_file)
            save_registry(registry)
            print("[+] Knowledge base completely indexed.")

        notify_progress(0.96, "Building BM25 keyword index...")
        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25_index = SimpleBM25(tokenized_corpus)
        notify_progress(1.0, f"Indexing Complete! Total chunks: {len(self.chunks)}.")

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

    def retrieve(self, question, k=3, use_hybrid=True, use_hyde=False, use_rerank=True, use_parent=True, allowed_namespaces=None):
        metrics = {}
        start = time.time()

        # Application scoping (Part 2): when a specific app is selected, restrict the
        # retrieval candidate pool to chunks whose namespace belongs to that app. This
        # stops another application's documents (e.g. its "user creation" steps) from
        # ever entering the context in the first place.
        allowed_set = None
        if allowed_namespaces:
            allowed_set = {str(ns).lower() for ns in allowed_namespaces}

        def _is_allowed(idx):
            if allowed_set is None:
                return True
            return self.chunks[idx].get("namespace", "").lower() in allowed_set

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
            # When filtering by namespace, pull a wider pool so enough in-scope chunks survive.
            search_k = min(self.faiss_index.ntotal, 100 if allowed_set else 20)
            _, s_indices = self.faiss_index.search(query_vec, k=search_k)
            semantic_ids = [int(idx) for idx in s_indices[0] if idx != -1 and _is_allowed(int(idx))][:20]
            metrics["semantic_time"] = time.time() - start
        else:
            semantic_ids = []
            metrics["semantic_time"] = 0.0

        # 2. Keyword Search (BM25)
        start = time.time()
        if use_hybrid:
            tokenized_query = tokenize(question)
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            ranked_ids = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            keyword_ids = [i for i in ranked_ids if _is_allowed(i)][:20]
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
                # Namespace boost: if question mentions the namespace, nudge the score
                try:
                    ns = candidates[i].get("namespace", "").lower()
                    if ns and ns in question.lower():
                        candidates[i]["score"] += 0.25
                except Exception:
                    pass
            candidates.sort(key=lambda x: x["score"], reverse=True)
        else:
            # If no reranking, scores are just their rank position
            for i, c in enumerate(candidates):
                c["score"] = 1.0 / (i + 1)
                try:
                    ns = c.get("namespace", "").lower()
                    if ns and ns in question.lower():
                        c["score"] += 0.25
                except Exception:
                    pass
        
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

    def generate_stream(self, question, context, history, max_tokens=512, api_key=None, app_prompt=None, is_small_talk_mode=False):
        # Application-scoping block. When the user has picked a specific application
        # (e.g. TrackWise, ThingWorx), this is prepended to the system prompt so that
        # ambiguous questions ("how is a user created") are answered strictly in the
        # context of that application instead of leaking details from another one.
        app_section = f"{app_prompt.strip()}\n\n" if app_prompt else ""

        def count_tokens(text):
            if self.is_gguf and hasattr(self.model, "tokenize"):
                try:
                    return len(self.model.tokenize(text.encode('utf-8', errors='ignore')))
                except Exception:
                    pass
            elif self.tokenizer:
                try:
                    return len(self.tokenizer.encode(text))
                except Exception:
                    pass
            return len(text) // 4

        # Perform context budgeting for local models to prevent context window overflow
        if not self.model_name.startswith("gemini-"):
            limit_n_ctx = self.n_ctx if self.is_gguf else 2048
            if not self.is_gguf and self.model and hasattr(self.model, "config"):
                limit_n_ctx = getattr(self.model.config, "max_position_embeddings", limit_n_ctx)
            
            # Estimate token usage of system prompt template without context
            system_template = app_section + (
                "IDENTITY AND CREATOR RULES:\n"
                "- Your name is CognIQ.\n"
                "- You are a local, secure closed-context corporate RAG assistant.\n"
                "- You were created and built by Cognizant.\n"
                "- You were specifically designed and developed for Fresenius Medical Care (FMC).\n"
                "- If the user asks about who you are, your creator, your developer, your name, your purpose, or the company you work for, you must answer immediately and professionally using the above details, bypassing the strict document context rule for these identity questions.\n\n"
                "GENERAL QA RULES:\n"
                "- For all other general and technical questions, you must answer using ONLY the provided context below.\n"
                "- If the context contains relevant information (even if it is an overview, summary, or partial description), use it to provide a helpful, comprehensive, and detailed answer. Describe whatever relevant details are present (such as key areas, contact names, tools, or overview steps).\n"
                "- Respond in a professional, corporate tone appropriate for an internal Fresenius Medical Care assistant.\n"
                "- You may relate the meanings of words in the question to the context to find the best matching information, but do not add any facts that are not explicitly present in the context.\n"
                "<context>\n\n</context>\n"
                "- Only if the provided context is completely unrelated or has zero connection to the user's question, reply exactly with: \"I think this info isn't yet added to my knowledge base.\" Do not add any explanations or extra words if you output this fallback phrase.\n"
                "- Ensure that your answers are complete and do not cut off mid-sentence. If the answer is long, provide it in full and do not truncate it. Always use all relevant information from the context to provide the most comprehensive answer possible."
            )
            
            skeleton_messages = [{"role": "system", "content": system_template}]
            for entry in history[-2:]:
                skeleton_messages.append({"role": "user", "content": entry["user"]})
                skeleton_messages.append({"role": "assistant", "content": entry["bot"]})
            skeleton_messages.append({"role": "user", "content": question})
            
            if self.is_gguf:
                skeleton_text = ""
                for m in skeleton_messages:
                    skeleton_text += f"{m['role']}: {m['content']}\n"
                skeleton_tokens = count_tokens(skeleton_text)
            else:
                if self.tokenizer:
                    try:
                        skeleton_text = self.tokenizer.apply_chat_template(skeleton_messages, tokenize=False, add_generation_prompt=True)
                        skeleton_tokens = count_tokens(skeleton_text)
                    except Exception:
                        skeleton_tokens = count_tokens(str(skeleton_messages))
                else:
                    skeleton_tokens = count_tokens(str(skeleton_messages))
            
            safety_headroom = max_tokens + 128
            max_context_tokens = max(512, limit_n_ctx - skeleton_tokens - safety_headroom)
            
            context_tokens = count_tokens(context)
            if context_tokens > max_context_tokens:
                print(f"[SYSTEM] Context size ({context_tokens} tokens) exceeds allowed budget ({max_context_tokens} tokens). Truncating context.")
                truncated_context = ""
                lines = context.split("\n")
                current_tokens = 0
                for line in lines:
                    line_tokens = count_tokens(line + "\n")
                    if current_tokens + line_tokens < max_context_tokens:
                        truncated_context += line + "\n"
                        current_tokens += line_tokens
                    else:
                        if not truncated_context:
                            char_limit = int(max_context_tokens * 3.5)
                            truncated_context = line[:char_limit] + "... [Truncated due to context limit]"
                        else:
                            truncated_context += "... [Truncated due to context limit]"
                        break
                context = truncated_context

        if is_small_talk_mode:
            system_msg = app_section + (
                "IDENTITY AND GREETING RULES:\n"
                "- Your name is CognIQ.\n"
                "- You are an internal corporate support AI assistant developed by Cognizant for Fresenius Medical Care (FMC).\n"
                "- The user is greeting you or starting a casual conversation (e.g., 'hi', 'hello', 'how are you', 'good morning', 'thanks').\n"
                "- Respond in a warm, polite, concise, and professional corporate tone (1-2 sentences maximum).\n"
                "- MANDATORY CONVERSATIONAL PIVOT: At the very end of your response, ALWAYS ask how you can help them today with their corporate support questions or specific applications (such as TrackWise, ThingWorx, Polarion, or Windchill GPDM), guiding them directly back to asking technical/support questions instead of continuing casual small talk."
            )
        else:
            system_msg = app_section + (
                "IDENTITY AND CREATOR RULES:\n"
                "- Your name is CognIQ.\n"
                "- You are a local, secure closed-context corporate RAG assistant.\n"
                "- You were created and built by Cognizant.\n"
                "- You were specifically designed and developed for Fresenius Medical Care (FMC).\n"
                "- If the user asks about who you are, your creator, your developer, your name, your purpose, or the company you work for, you must answer immediately and professionally using the above details, bypassing the strict document context rule for these identity questions.\n\n"
                "GENERAL QA RULES:\n"
                "- For all other general and technical questions, you must answer using ONLY the provided context below.\n"
                "- If the context contains relevant information (even if it is an overview, summary, or partial description), use it to provide a helpful, comprehensive, and detailed answer. Describe whatever relevant details are present (such as key areas, contact names, tools, or overview steps).\n"
                "- Respond in a professional, corporate tone appropriate for an internal Fresenius Medical Care assistant.\n"
                "- You may relate the meanings of words in the question to the context to find the best matching information, but do not add any facts that are not explicitly present in the context.\n"
                f"<context>\n{context}\n</context>\n"
                "- Only if the provided context is completely unrelated or has zero connection to the user's question, reply exactly with: \"I think this info isn't yet added to my knowledge base.\" Do not add any explanations or extra words if you output this fallback phrase.\n"
                "- Ensure that your answers are complete and do not cut off mid-sentence. If the answer is long, provide it in full and do not truncate it. Always use all relevant information from the context to provide the most comprehensive answer possible."
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

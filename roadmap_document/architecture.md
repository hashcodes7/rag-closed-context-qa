# 🏛️ SourceIQ: End-to-End System Architecture Blueprint (v20)

This document provides a highly detailed, comprehensive, top-to-bottom architectural blueprint of **SourceIQ (Version 20)**, a state-of-the-art local Retrieval-Augmented Generation (RAG) chatbot application. SourceIQ balances semantic understanding, high-performance local model execution (GGUF / Quantized Transformers), and rigorous context grounding.

---

## 🗺️ High-Level System Architecture

Below is a complete, top-to-bottom data flow and control architecture diagram mapping both the **Offline Ingestion & Indexing Pipeline** (how files are processed and structured) and the **Online Retrieval & Generation Pipeline** (how questions are resolved, fused, reranked, and answered).

```mermaid
graph TD
    %% Define Styles %%
    classDef ui fill:#302b63,stroke:#ffffff,stroke-width:2px,color:#ffffff;
    classDef process fill:#1e1e30,stroke:#3b5998,stroke-width:1px,color:#ffffff;
    classDef storage fill:#0f0c29,stroke:#00d2ff,stroke-width:2px,color:#ffffff;
    classDef ai fill:#24243e,stroke:#f000ff,stroke-width:2px,color:#ffffff;

    subgraph OFFLINE_INGESTION ["Offline Ingestion & Indexing Pipeline"]
        A[knowledge_source folder] -->|File Ingestion| B(Document Extractor: txt, pdf, docx, html)
        B -->|Raw Text| C{Chunking Strategy Toggle}
        C -->|Semantic| D(Semantic Chunking: Embeddings + Cosine Sim Breakpoint)
        C -->|Recursive| E(Recursive Character Chunking)
        D & E -->|Parent Chunks| F(Parent-Child Context Mapper)
        F -->|Large Parent Chunks| G[(parent_chunks Registry)]
        F -->|Small Overlapping Child Chunks| H(SentenceTransformer Embedder)
        H -->|Dense Vector Embeddings| I(FAISS HNSW Indexer)
        I -->|Write Index File| J[(faiss_index.bin)]
        G & H -->|Write Cache File| K[(vector_cache.pt)]
        H -->|Tokenized Corpus| L(SimpleBM25 Sparse Indexer)
    end

    subgraph ONLINE_QUERY ["Online Retrieval & Generation Pipeline"]
        M[Streamlit UI Chat Input] -->|User Question| N{HyDE Toggle}
        N -->|Enabled| O(Zero-Shot Query Expansion LLM)
        O -->|Hypothetical Answer| P[Expanded Search Query]
        N -->|Disabled| P
        
        P -->|Query Vectorization| Q(SentenceTransformer Embedder)
        Q -->|Dense Vector| R(Semantic Search: FAISS HNSW)
        P -->|Tokenized Query| S(Keyword Search: BM25)
        
        R -->|Top-N Semantic Chunks| T{Hybrid Search Toggle}
        S -->|Top-N Keyword Chunks| T
        
        T -->|Enabled| U(Reciprocal Rank Fusion RRF)
        T -->|Disabled| V[Semantic Chunks Only]
        U & V -->|Candidates List| W{Reranking Toggle}
        
        W -->|Enabled| X(Cross-Encoder Rerank Model)
        W -->|Disabled| Y[Ranked Candidates]
        X -->|Reranked Candidates| Y
        
        Y -->|Top-K Selected Chunks| Z{Parent-Doc Toggle}
        Z -->|Enabled| AA(Child-to-Parent Lookup: parent_chunks Mapping)
        Z -->|Disabled| AB[Use Child Text Directly]
        
        AA & AB -->|Context Payload| AC(System Prompt Construction)
        AD[(SQLite rag_history.db)] <-->|Load/Save History| AE(Rolling Chat History Buffer)
        AE -->|Last 2 Turns Context| AC
        
        AC -->|Assembled Prompt| AF{Model Engine Mode}
        AF -->|GGUF Mode| AG(llama-cpp-python Inference Engine)
        AF -->|Transformers Mode| AH(Hugging Face BitsAndBytes Quantized LLM)
        
        AG & AH -->|Streaming Text Generation| AI[Streamlit Real-Time Visualizer]
        AI -->|Log Telemetry & Citation Verification| AJ[(SQLite DB / UI Telemetry Charts)]
    end

    %% Applying classes %%
    class A,M,AI,AJ ui;
    class B,C,D,E,F,H,I,L,O,Q,R,S,U,X,AA,AC,AE,AF,AG,AH process;
    class G,J,K,AD storage;
```

---

## 🧩 Deep Dive Component Explanations

Each component in SourceIQ has a specialized role. Below, we break down every stage in the order of execution, detailing **What** it is, **Why** we do it, and **How** it is implemented in the codebase.

---

### 1. Document Extraction & Preprocessing

```
[knowledge_source] ──> [Extracted Text] (txt, pdf, docx, html)
```

#### 📖 What is the step in detail?
This is the starting point of the ingestion pipeline. SourceIQ monitors the `knowledge_source` folder for multiple file formats (`.txt`, `.pdf`, `.docx`, `.html`, `.htm`). When an ingestion trigger is fired, the pipeline scans this folder, detects the extensions, bypasses non-textual data, and extracts the raw textual content. It handles different text encodings (defaulting to `UTF-8`) and scrubs formatting tags or document metadata.

#### 🎯 Why are we doing that step?
Knowledge in a production environment is rarely stored in clean, plain text. By integrating native extractors:
- Users can drop high-fidelity documents (e.g., standard manuals, web pages, academic reports) directly into the folder without manual conversion.
- Eliminates overhead by stripping unnecessary HTML syntax or document styling, ensuring only semantic text is indexed, which reduces token usage and improves search relevance.

#### 🛠️ How are we doing it?
Implemented in `core.py` under the function `extract_text_from_file(filepath)`:
- **`.txt`**: Handled via native Python `open()` with a strict `utf-8` decoding fallback.
- **`.pdf`**: Handled via **PyMuPDF** (`fitz`). It iterates through every page in the document and extracts the text with coordinate separation:
  ```python
  import fitz
  doc = fitz.open(filepath)
  text = "".join([page.get_text() + "\n" for page in doc])
  ```
- **`.docx`**: Handled via **python-docx** (`docx`), joining paragraph runs:
  ```python
  import docx
  doc = docx.Document(filepath)
  text = "\n".join([para.text for para in doc.paragraphs])
  ```
- **`.html` / `.htm`**: Handled via **BeautifulSoup** (`bs4`), removing script and style blocks before harvesting the textual tree:
  ```python
  from bs4 import BeautifulSoup
  soup = BeautifulSoup(f, "html.parser")
  for script in soup(["script", "style"]):
      script.extract()
  text = soup.get_text(separator=' ')
  ```

---

### 2. Semantic Chunking Strategy (Adaptive Partitioning)

```
[Full Text] ──> [Embeddings per Sentence] ──> [Breakpoint Analysis (Threshold: 0.5)] ──> [Parent Chunks]
```

#### 📖 What is the step in detail?
Traditional chunking splits text at fixed character limits, which often breaks paragraphs mid-sentence. **Semantic Chunking** is an advanced strategy that splits text based on the semantic shift of ideas.
1. The raw text is tokenized into individual sentences.
2. Every sentence is converted into an embedding using a vector model.
3. The cosine similarity between each consecutive sentence ($S_{i}$ and $S_{i-1}$) is calculated.
4. When similarity drops below a predefined threshold (e.g., `0.5`), it represents a logical breakpoint or shift in topic, triggering the creation of a new **Parent** chunk. A size cap (e.g., `1200` characters) prevents individual chunks from ballooning.

#### 🎯 Why are we doing that step?
- **Context Integrity**: Prevents "context fragmentation," where a crucial sentence is split in half across two chunks, rendering both chunks useless or confusing to the LLM.
- **Better Retrieval Quality**: Guarantees that each retrieved chunk represents a coherent, complete idea or topic, leading to cleaner retrieval and higher-quality generation.

#### 🛠️ How are we doing it?
Implemented in `core.py` in the `semantic_chunk_text(text, embedder, threshold=0.5, max_chunk_size=1200)` function:
- Replaces multiple newlines with periods to regularize sentence splits.
- Split sentences using regex lookbehinds: `re.split(r'(?<=[.!?]) +', clean_text)`.
- Generates sentence embeddings in batches:
  ```python
  embeddings = embedder.encode(sentences, batch_size=64, convert_to_numpy=True)
  ```
- Iterates over sentences, computing the normalized inner product (cosine similarity) between consecutive vectors:
  ```python
  sim = np.dot(embeddings[i], embeddings[i-1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]) + 1e-9)
  ```
- Creates a new chunk if `sim < threshold` or if the active paragraph size exceeds `max_chunk_size`.

---

### 3. Parent-Child Context Mapping

```
Parent Chunk (e.g., 1200 chars)
  ├── Child Chunk 1 (400 chars, 50 overlap)  ──> Vector Search Index
  └── Child Chunk 2 (400 chars, 50 overlap)  ──> Vector Search Index
```

#### 📖 What is the step in detail?
This solves the **Precision vs. Context window** trade-off. 
- Large chunks are great for providing the LLM with background context, but their embedding vectors are highly diffused, making them hard to locate using short user questions.
- Small chunks contain highly focused vectors that are easily matched, but they lack the surrounding context the LLM needs to answer complex questions.

SourceIQ implements a **Parent-Child** mapping structure:
1. **Extraction**: The text is first split into broad **Parent** chunks (via Semantic Chunking or large recursive chunks).
2. **Deconstruction**: Each Parent chunk is mapped to a unique ID (e.g., `p_0`) and saved in a central registry. The Parent is then broken down into smaller, overlapping **Child** chunks (typically `400` characters with `50` overlap).
3. **Indexing**: Only the small **Child** chunks are embedded and indexed in the vector database.
4. **Retrieval**: The vector search locates the high-precision Child chunk, but instead of feeding this snippet directly to the LLM, the system references the `parent_id` and pulls the larger **Parent** chunk to serve as context.

#### 🎯 Why are we doing that step?
- Allows the search engine to perform highly specific matching. A single specific sentence in a manual can be retrieved instantly.
- Saves the LLM from trying to answer questions using disjointed, fragmented fragments. By expanding back to the Parent chunk, the LLM receives the complete paragraph, retaining the necessary context.

#### 🛠️ How are we doing it?
Implemented in `core.py` inside `process_knowledge_base()`:
- Parent chunks are stored in `self.parent_chunks = {}` mapped to unique string keys (`f"p_{parent_id_counter}"`).
- Child chunks are generated from the parents using `recursive_chunk_text(p_text, chunk_size=400, overlap=50)` and stored in a list of dicts:
  ```python
  self.chunks.append({
      "source": filename,
      "chunk_id": i,
      "text": c_text,
      "parent_id": p_id
  })
  ```
- During search, `retrieve()` checks the `use_parent` toggle:
  ```python
  if use_parent:
      p_id = item.get("parent_id")
      item["retrieval_text"] = item["text"] # Retained for UI citation highlight
      item["text"] = self.parent_chunks[p_id] # Injected into LLM context payload
  ```

---

### 4. Semantic Search (FAISS HNSW Vector Index)

```
[Query Text] ──> [SentenceTransformer] ──> [Normalized Vector] ──> [FAISS HNSW Inner Product Search]
```

#### 📖 What is the step in detail?
This is the **dense retrieval** engine. Dense retrieval translates text into high-dimensional numerical vectors where distance correlates with semantic similarity. Rather than doing a brute-force cosine comparison against every chunk in the database (which slows down as documents grow), SourceIQ uses a **Hierarchical Navigable Small Worlds (HNSW)** graph index. HNSW constructs a multi-layered graph where the top layer has long-range links for fast routing, and the bottom layer has short-range links for exact nearest-neighbor search.

#### 🎯 Why are we doing that step?
- **Speed**: HNSW provides logarithmic search time complexity $O(\log N)$, enabling sub-millisecond retrieval speeds even across millions of chunks.
- **Semantic Understanding**: Handles synonyms, paraphrasing, and conceptually related language where keyword matching would completely fail.

#### 🛠️ How are we doing it?
Implemented in `core.py` using **FAISS** (`faiss`) and **SentenceTransformers**:
- Vectorization utilizes the `all-MiniLM-L6-v2` model, yielding a 384-dimensional dense space.
- Vectors are normalized to unit length ($L_2$ normalization) so that a simple inner product matches cosine similarity:
  ```python
  embeddings_np = self.embedder.encode(chunk_texts, convert_to_numpy=True).astype("float32")
  faiss.normalize_L2(embeddings_np)
  ```
- Creates an HNSW index with standard construction hyperparameters ($M=32$ bidirectional links, and an construction search depth $efConstruction=200$):
  ```python
  dim = embeddings_np.shape[1]
  self.faiss_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
  self.faiss_index.hnsw.efConstruction = 200
  self.faiss_index.add(embeddings_np)
  ```
- Query-time search controls search depth via $efSearch = 64$ to balance accuracy and speed:
  ```python
  self.faiss_index.hnsw.efSearch = 64
  _, s_indices = self.faiss_index.search(query_vec, k=20)
  ```

---

### 5. Persistent Vector Caching

```
        ┌──> Store as torch serialization ──> [vector_cache.pt]
Ingest ─┤
        └──> Store HNSW binary graph ───────> [faiss_index.bin]
```

#### 📖 What is the step in detail?
Running deep-learning embedding models over thousands of sentences is computationally heavy. **Persistent Vector Caching** ensures that the indexing phase happens **exactly once**. Once the knowledge base is scanned and vectorized, the raw text database, parent mappings, and FAISS index are dumped to local binary files on disk. On subsequent startups, the engine compares the set of files present in `knowledge_source` with the cache metadata. If they match, the vector embeddings are loaded instantly from disk.

#### 🎯 Why are we doing that step?
- **Instant Boot Times**: Reduces startup time from minutes (vectorizing thousands of paragraphs) to less than a second on CPU.
- **Saves Resources**: Eliminates redundant GPU/CPU embedding cycles when files haven't changed.
- **Graceful Sync**: The system auto-detects if files are added, modified, or deleted and triggers a rebuild only when necessary.

#### 🛠️ How are we doing it?
Implemented in `core.py` under `process_knowledge_base()`:
- Writes the chunk registry and parent dictionary using PyTorch's serialization mechanism to `vector_cache.pt`:
  ```python
  torch.save({"chunks": self.chunks, "parent_chunks": self.parent_chunks}, cache_file)
  ```
- Writes the FAISS graph layout to `faiss_index.bin`:
  ```python
  faiss.write_index(self.faiss_index, index_file)
  ```
- Compares the file checklist at startup to check if a rebuild is needed:
  ```python
  disk_files = {f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)}
  data = torch.load(cache_file)
  cached_files = {c["source"] for c in data.get("chunks", [])}
  if disk_files == cached_files:
      # Load cache instantly
  ```

---

### 6. Keyword Search (Custom BM25)

```
[Query Token list] ──> [Lookup term IDF & Doc frequencies] ──> [BM25 Scoring Formula]
```

#### 📖 What is the step in detail?
This is the **sparse retrieval** engine. Dense embeddings are brilliant at understanding meaning, but they fail at matching exact numbers, product serial keys, unique error codes, or rare named entities (e.g. searching for a specific ID like `"XAE-908"`). Keyword search parses text into term frequencies ($tf$) and inverse document frequencies ($idf$). SourceIQ implements the classic **BM25** ranking algorithm, which scales log-linearly with term occurrences and applies doc-length normalization.

#### 🎯 Why are we doing that step?
- **Hybrid Balance**: Acts as a safety net for FAISS. When a user asks for an exact phrase, rare parameter name, or code symbol, BM25 ensures that the exact document is retrieved, even if it is conceptually distant from the embedding's semantic space.

#### 🛠️ How are we doing it?
Implemented as a native, lightweight class `SimpleBM25` in `core.py`:
- Ingestion tokenizes text by stripping punctuation and converting to lowercase:
  ```python
  def tokenize(text):
      return re.sub(r'[^\w\s]', ' ', text.lower()).split()
  ```
- Precomputes Inverse Document Frequency ($IDF$) for every word across the corpus:
  $$IDF(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$$
- Scores candidate documents by checking terms against term frequency ($f(q_i, D)$), normalized by document length ($|D|$) vs average document length ($avgdl$):
  ```python
  for i in range(self.corpus_size):
      fi = self.doc_freqs[i].get(word, 0)
      scores[i] += idf * (fi * (self.k1 + 1)) / (fi + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
  ```
  *(Where hyperparameters are set to standard defaults: $k_1 = 1.5$ for term frequency saturation, and $b = 0.75$ for document length penalization).*

---

### 7. Reciprocal Rank Fusion (RRF)

```
Rankings: FAISS (1st, 2nd, 3rd) & BM25 (1st, 2nd, 3rd) 
   ── RRF Formula [ 1 / (60 + Rank) ] ──> Blended & Re-ordered Candidate List
```

#### 📖 What is the step in detail?
When combining Keyword search (BM25) and Semantic search (FAISS), you run into a problem: their scores are in completely different ranges. FAISS outputs inner product distances (e.g., `0.3` to `0.9`), while BM25 outputs unbounded positive score frequencies (e.g., `1.2` to `25.0`).
**Reciprocal Rank Fusion (RRF)** is a robust, scale-free rank aggregation method that combines multiple search lists by summing their reciprocal positions:
$$RRF\_Score(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
Where $r_m(d)$ is the rank of document $d$ in the system $m$, and $k$ is a constant smoothing parameter (typically `60`).

#### 🎯 Why are we doing that step?
- **Unification**: Blends dense and sparse search results without needing to normalize scores or tune fragile scaling coefficients.
- **Top-Rank Safety**: Ensures that if a document is ranked highly in *either* keyword or semantic search, it remains near the top of the final merged candidate list.

#### 🛠️ How are we doing it?
Implemented in `core.py` in `reciprocal_rank_fusion(results_list, k=60)`:
- Maps all candidate IDs and aggregates their reciprocal rank:
  ```python
  fused_scores = {}
  for results in results_list:
      for rank, idx in enumerate(results):
          fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank)
  sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
  ```

---

### 8. HyDE (Hypothetical Document Embeddings - Query Expansion)

```
User Query ──> [LLM Direct Prompt] ──> Hypothetical Answer ──> [Combine: Query + Answer] ──> Vector Search
```

#### 📖 What is the step in detail?
User questions are typically short and phrased as queries, whereas target documents are written in an expository, declarative style. This mismatch can lead to poor embedding alignments.
**HyDE** bypasses this gap:
1. When the user asks a question, the LLM is first used to generate a brief **hypothetical answer** (even if it's factually inaccurate or contains hallucinations).
2. The user's query and this hypothetical answer are merged together.
3. This merged text is then embedded and used to query the FAISS index.

#### 🎯 Why are we doing that step?
- **Conceptual Bridging**: The hypothetical answer matches the writing style, vocabulary, and semantic tone of the target documents much better than a raw question.
- **Drastically Improves Sparse Queries**: Incredibly helpful for short, conceptual questions where standard semantic search might get sidetracked by keyword differences.

#### 🛠️ How are we doing it?
Implemented in `core.py` in `generate_hypothetical_answer(question)` and integrated into `retrieve()`:
- Calls the loaded model using a one-sentence directive: `"Write a one-sentence technical answer to this question: {question}"`.
- Extracts the output, appends it to the query, and routes it to the embedding pipeline:
  ```python
  if use_hyde:
      hyde_answer = self.generate_hypothetical_answer(question)
      search_query = f"{question} {hyde_answer}"
  ```

---

### 9. Cross-Encoder Reranking

```
Merged Candidates ──> [Cross-Encoder Model] (Interprets Query + Chunk simultaneously) ──> Final Filtered Top-K
```

#### 📖 What is the step in detail?
A bi-encoder vector model (used in the initial search phase) calculates embeddings for the query and documents independently, then calculates their similarity. This is extremely fast but misses complex, fine-grained details between the query and text.
A **Cross-Encoder** reranker does not embed documents separately. Instead, it feeds the user's query and the candidate document chunk *together* into a transformer, allowing the attention mechanism to analyze the interactions between every single word in both texts.

#### 🎯 Why are we doing that step?
- **Accuracy Boost**: Acts as the ultimate precision filter. While bi-encoder search is fast and great for pulling the top 20 candidates, the Cross-Encoder acts as a careful editor, filtering out irrelevant chunks that happened to have similar-looking keywords but didn't actually answer the question.

#### 🛠️ How are we doing it?
Implemented in the `retrieve()` routine in `core.py`:
- In v20, the Cross-Encoder can be toggled on/off in the Streamlit UI depending on resources.
- If enabled, the system feeds the candidate chunks to the reranker model (`ms-marco-MiniLM-L-6-v2`) to score the `[query, document]` pairs directly:
  ```python
  if use_rerank and self.cross_encoder is not None:
      cross_inp = [[question, item["text"]] for item in candidates]
      cross_scores = self.cross_encoder.predict(cross_inp)
      for i in range(len(candidates)):
          candidates[i]["score"] = float(cross_scores[i])
      candidates.sort(key=lambda x: x["score"], reverse=True)
  ```

---

### 10. Quantized Inference Execution (GGUF vs Transformers)

```
                    ┌──> Standard PyTorch ──> BitsAndBytes (4-bit/8-bit Quantized) 
Engine Selection ───┤
                    └──> CPU Optimized GGUF ──> llama-cpp-python (C++ High Perf)
```

#### 📖 What is the step in detail?
A major bottleneck of local RAG is LLM execution on consumer computers. SourceIQ features a dual-execution engine that supports both standard PyTorch transformers models and C++ optimized GGUF engines. GGUF quantizes model weights to 4-bit or 8-bit, drastically reducing memory usage and making it possible to run high-quality LLMs on standard CPUs without requiring expensive graphics cards.

#### 🎯 Why are we doing that step?
- **Hardware Agnostic**: Ensures standard laptops can run the system locally at decent speeds.
- **Resource Efficiency**: Quantization reduces memory usage by up to 75% while keeping performance drop to a minimum.

#### 🛠️ How are we doing it?
Implemented in `core.py` in `load_models()` and `generate_stream()`:
- **GGUF Mode**: Leverages `llama-cpp-python` with lazy GPU layers, auto-detecting core count for maximum CPU threading performance:
  ```python
  self.model = Llama(
      model_path=gguf_file, 
      n_ctx=2048, 
      n_threads=os.cpu_count() or 4,
      n_gpu_layers=-1 if torch.cuda.is_available() else 0
  )
  ```
- **Transformers Mode**: Uses **BitsAndBytes** to load standard HF PyTorch weights in 4-bit NormalFloat (NF4) quantization:
  ```python
  quant_config = BitsAndBytesConfig(
      load_in_4bit=True, 
      bnb_4bit_compute_dtype=torch.float16, 
      bnb_4bit_quant_type="nf4", 
      bnb_4bit_use_double_quant=True
  )
  self.model = AutoModelForCausalLM.from_pretrained(
      self.model_name, 
      quantization_config=quant_config
  )
  ```

---

### 11. Strict Context Grounding & Generation

```
[System Prompt + Strict Guidelines] ──> [Chat History] ──> [Context Payload] ──> LLM Stream Output
```

#### 📖 What is the step in detail?
Once retrieval is finished, the system builds the generation prompt. The prompt prepends a highly descriptive, strict system instruction that acts as a guardrail. The retrieved context chunks are loaded into XML-like tags, and the recent conversation history (last 2 turns) is appended for conversational awareness.
The LLM is prompted to answer the question using **ONLY** the provided context. If the context does not contain the answer, it must output a specific fallback phrase, preventing the model from hallucinating or using its pre-trained general knowledge.

#### 🎯 Why are we doing that step?
- **Eliminates Hallucinations**: Enforces strict closed-context QA. If a fact isn't in the provided documentation, the engine won't guess.
- **Reliable Citations**: By strictly binding the answer to the provided context segments, SourceIQ can trace every generated statement back to a specific document page and paragraph.

#### 🛠️ How are we doing it?
Implemented in `core.py` in `generate_stream()`:
- Formulates a strict, clear prompt template:
  ```
  You are a strict, factual question-answering assistant. Your task is to answer the user's question using ONLY the provided Context below. Do not assume, extrapolate, or bring in outside knowledge.
  --- CONTEXT START ---
  {context}
  --- CONTEXT END ---
  CRITICAL INSTRUCTIONS:
  1. If the Context contains the answer, provide a concise, direct response based exactly on the text.
  2. If the Context does NOT contain the answer... state exactly: 'I cannot answer this question based on the provided context.'
  ```
- Assembles chat history in standard ChatML/Instruction dictionaries:
  ```python
  messages = [{"role": "system", "content": system_msg}]
  for entry in history[-2:]: # Last 2 turns
      messages.append({"role": "user", "content": entry["user"]})
      messages.append({"role": "assistant", "content": entry["bot"]})
  messages.append({"role": "user", "content": question})
  ```
- Uses a background thread and a `TextIteratorStreamer` to yield tokens on-the-fly, creating a real-time typing effect in the UI.

---

### 12. SQLite Telemetry & Persistence Layer

```
Streamlit Action ──> [Save/Read SQL] ──> [rag_history.db] (Relational Tables)
```

#### 📖 What is the step in detail?
A professional assistant needs a memory that lasts longer than a single browser refresh. SourceIQ includes an SQLite database persistence layer (`rag_history.db`). Every interaction—including user prompts, assistant replies, source citations, and pipeline metrics (e.g. how many milliseconds were spent on semantic search, keyword lookup, HyDE expansion, or generation)—is logged to a local relational database.

#### 🎯 Why are we doing that step?
- **Continuous Memory**: Restores past chat logs instantly when the UI is refreshed or restarted.
- **Telemetry Charts**: Allows the Streamlit UI to render real-time performance graphs, showing users which parts of the pipeline are taking the most time and letting them optimize their pipeline settings.

#### 🛠️ How are we doing it?
Implemented in `database.py` and integrated into `ui.py`:
- Initializes an SQLite schema with tables for `session_id`, `role`, `content`, JSON serialized `sources`, and JSON serialized `metrics`:
  ```sql
  CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      role TEXT,
      content TEXT,
      sources TEXT,
      metrics TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
  )
  ```
- Saves interaction logs to the database using parameters to prevent SQL injection:
  ```python
  db.save_message("default_user", "assistant", response, sources_meta, metrics)
  ```
- Loads past messages at startup via `db.load_messages("default_user")`.

---

## 📈 Summary of Offline vs. Online Pipeline Boundaries

To recap how data flows across the system boundary, here is the distinct operational split:

| Feature / Step | Processing Mode | Major Libraries / Files | Core Purpose |
| :--- | :--- | :--- | :--- |
| **Ingestion** | Offline (or Manual Trigger) | `core.py`, PyMuPDF, python-docx, bs4 | Scans `knowledge_source/` and extracts raw text. |
| **Chunking** | Offline (or Manual Trigger) | `core.py` (`semantic_chunk_text`) | Segments documents using sentence semantic boundaries. |
| **Parent Mapping** | Offline (or Manual Trigger) | `core.py` (`process_knowledge_base`) | Builds Parent registry and splits children for dense search. |
| **FAISS Indexing** | Offline (or Manual Trigger) | `core.py`, FAISS, SentenceTransformer | Vectorizes child chunks and builds an HNSW graph. |
| **Caching** | Offline (or Manual Trigger) | `core.py`, PyTorch | Saves embeddings to `vector_cache.pt`/`faiss_index.bin`. |
| **HyDE Query Expansion** | Online (Interactive) | `core.py` (`generate_hypothetical_answer`) | Generates a fake answer to improve search matching. |
| **Hybrid Search** | Online (Interactive) | `core.py` (FAISS HNSW + `SimpleBM25`) | Merges semantic and keyword matching parallel streams. |
| **Aggregation (RRF)** | Online (Interactive) | `core.py` (`reciprocal_rank_fusion`) | Blends ranks from multiple retrieval techniques. |
| **Reranking** | Online (Interactive) | `core.py`, Cross-Encoder | Filters and refines the top retrieved candidates. |
| **Context Assembly** | Online (Interactive) | `core.py` (`retrieve` Parent lookup) | Swaps child search matches with full parent paragraphs. |
| **Generation** | Online (Interactive) | `core.py` (`generate_stream`), llama.cpp | Streams answers under strict grounding prompts. |
| **Telemetry & Log** | Online (Interactive) | `database.py`, Streamlit (`ui.py`) | Persists history and visualizes processing times in charts. |

---
© 2026 SourceIQ Engineering | *Modular RAG Systems Design Group*

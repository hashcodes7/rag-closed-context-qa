import streamlit as st
import time
import os
from core import RAGEngine, extract_text_from_file
import database as db

# Initialize database
db.init_db()

# =====================================================
def download_model_ui(repo_id, pattern="q4_k_m.gguf"):
    import huggingface_hub
    import requests
    
    try:
        info = huggingface_hub.model_info(repo_id)
    except Exception as e:
        st.error(f"Failed to fetch model info from Hugging Face: {e}")
        return None
        
    filename = None
    for sibling in info.siblings:
        if pattern.lower() in sibling.rfilename.lower():
            filename = sibling.rfilename
            break
            
    if not filename:
        st.error(f"Could not find a matching GGUF file in {repo_id}")
        return None
        
    url = huggingface_hub.hf_hub_url(repo_id, filename)
    os.makedirs("models", exist_ok=True)
    local_path = os.path.join("models", f"{repo_id.replace('/', '_')}_{filename}")
    
    st.write(f"📥 Downloading `{filename}` from `{repo_id}`...")
    
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        downloaded = 0
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size
                        progress_bar.progress(min(1.0, pct))
                        status_text.text(f"Downloaded {downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB ({pct:.1%})")
                        
        status_text.success(f"Download complete! Saved to `{local_path}`")
        return local_path
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[!] Download failed:\n{error_trace}")
        st.error(f"Download failed: {e}\n\n```python\n{error_trace}\n```")
        return None

# =====================================================
# 🌊 AskBot (Streamlit UI)
# =====================================================

st.set_page_config(page_title="AskBot", page_icon="🧠", layout="wide")

# --- INITIALIZE SESSION STATE ---
# --- INITIALIZE SESSION STATE ---
if "current_model" not in st.session_state:
    st.session_state["current_model"] = "bartowski/Llama-3.2-1B-Instruct-GGUF"

# --- CACHED ENGINE INITIALIZATION ---
@st.cache_resource
def get_engine(model_name):
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    return RAGEngine(model_name, embed_model_name, cross_encoder_model_name)

engine = get_engine(st.session_state["current_model"])

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.title("⚙️ Engine Settings")
    st.divider()
    
    st.subheader("🤖 Model Selection")
    
    model_mapping = {
        "Meta Llama 3.2 1B Instruct GGUF": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "Google Gemini 1.5 Flash": "gemini-1.5-flash",
        "Google Gemini 1.5 Pro": "gemini-1.5-pro",
        "Microsoft Phi-3.5 Mini Instruct GGUF": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "Microsoft Phi-2 GGUF": "TheBloke/phi-2-GGUF",
        "Custom Model...": "Custom Model..."
    }
    
    display_options = list(model_mapping.keys())
    
    # Find the display name for the current model in session state
    current_display = "Meta Llama 3.2 1B Instruct GGUF"
    for disp, raw in model_mapping.items():
        if raw == st.session_state.get("current_model"):
            current_display = disp
            break
            
    # Handle case where current model is a custom input not in mapping
    if current_display == "Meta Llama 3.2 1B Instruct GGUF" and st.session_state.get("current_model") and st.session_state["current_model"] not in model_mapping.values():
        current_display = "Custom Model..."
            
    selected_display = st.selectbox("Choose a model", display_options, 
                                 index=display_options.index(current_display))
                                 
    selected_base = model_mapping.get(selected_display, selected_display)
    
    # 🆕 GGUF vs API vs Standard Indicators
    if selected_base.startswith("gemini-"):
        st.success("☁️ **Google Gemini API**")
        st.caption("Fast and powerful cloud inference.")
        api_key = st.text_input("Google API Key", type="password")
        st.session_state["google_api_key"] = api_key
    else:
        st.session_state["google_api_key"] = None
        is_gguf = "gguf" in selected_base.lower() or selected_base.endswith(".gguf")
        if is_gguf:
            st.success("⚡ **GGUF (Fast CPU Mode)**")
            st.caption("Running highly optimized C++ inference.")
        else:
            st.info("🌐 **Standard (Normal Mode)**")
            st.caption("Running standard Transformers inference.")
    
    final_model_name = selected_base
    if selected_base == "Custom Model...":
        custom_name = st.text_input("Enter HF Model ID", value=st.session_state["current_model"] if st.session_state["current_model"] not in model_options else "")
        if custom_name:
            final_model_name = custom_name

    # Model Switch Logic
    if st.session_state["current_model"] != final_model_name:
        if st.button("🚀 Apply Model Switch", use_container_width=True):
            st.session_state["current_model"] = final_model_name
            st.cache_resource.clear()
            st.session_state.pop("models_loaded", None) # Force re-load
            st.rerun()

    st.divider()
    st.subheader("🛠️ Algorithm Control")
    use_hybrid = st.toggle("Hybrid Search (BM25)", value=True, help="Combines keyword search with vector search.")
    use_hyde = st.toggle("HyDE Expansion", value=False, help="Generates an ideal answer first to improve retrieval.")
    use_rerank = st.toggle("Cross-Encoder Rerank", value=True, help="Uses a secondary model to refine result relevance.")
    use_parent = st.toggle("Parent-Doc Context", value=True, help="Retrieves the full paragraph context for the LLM.")
    
    st.divider()
    quant_mode = st.selectbox("Quantization Mode", ["4bit", "8bit", "full"], index=0)
    chunking_mode = st.radio("Chunking Strategy", ["semantic", "recursive"], index=0)
    
    st.divider()
    st.subheader("📂 Knowledge Manager")
    
    # File Uploader
    uploaded_files = st.file_uploader("Upload Knowledge Files", type=["txt", "pdf", "docx", "html"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join("knowledge_source", uploaded_file.name)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded: {uploaded_file.name}")
                st.session_state["reindex_required"] = True

    # File List & Deletion
    st.write("Current Files:")
    if os.path.exists("knowledge_source"):
        # Supported extensions should mirror core's accepted list
        supported_exts = {".txt", ".pdf", ".docx", ".html", ".htm"}

        # Determine which files are already indexed (if the engine has processed the KB)
        try:
            indexed_sources = set([c["source"] for c in engine.chunks]) if getattr(engine, "chunks", None) else set()
        except Exception:
            indexed_sources = set()

        for f in os.listdir("knowledge_source"):
            ext = os.path.splitext(f)[1].lower()

            # Status marker: ✓ indexed, ✗ unsupported, ○ supported-but-not-indexed
            if ext not in supported_exts:
                status = "✗"
                status_title = "Unsupported file type"
            elif f in indexed_sources:
                status = "✓"
                status_title = "Indexed"
            else:
                status = "○"
                status_title = "Supported (not indexed)"

            icon = "📄" if ext == ".txt" else "📕" if ext == ".pdf" else "📘" if ext == ".docx" else "🌐" if ext in {".html", ".htm"} else "📁"
            col_file, col_del = st.columns([0.8, 0.2])
            col_file.caption(f"{status} {icon} {f}")
            col_file.write(f"_{status_title}_")
            if col_del.button("🗑️", key=f"del_{f}"):
                os.remove(os.path.join("knowledge_source", f))
                st.session_state["reindex_required"] = True
                st.rerun()
                    
    if st.session_state.get("reindex_required"):
        st.warning("⚠️ Files changed. Re-index recommended.")
        
    if st.button("🛠️ Force Re-index Knowledge Base", use_container_width=True):
        with st.status("🏗️ Re-indexing...", expanded=True) as status:
            engine.process_knowledge_base(force_reindex=True, chunking_mode=chunking_mode)
            st.session_state["reindex_required"] = False
            status.update(label="✅ Re-indexed Successfully!", state="complete", expanded=False)
            st.rerun()
    
    st.divider()
    if st.button("🔄 Force Engine Restart"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

    if st.button("🗑️ Clear Chat History"):
        db.clear_history("default_user")
        st.session_state["messages"] = []
        st.rerun()

    st.info("AskBot - Advanced Edition")

# --- INITIALIZE MODELS & DATA ---
if "reindex_required" not in st.session_state:
    st.session_state["reindex_required"] = False

if "models_loaded" not in st.session_state:
    repo_id = st.session_state["current_model"]
    
    is_cached = False
    if repo_id.startswith("gemini-") or repo_id == "Custom Model...":
        is_cached = True
    else:
        if os.path.exists("models"):
            target_prefix = repo_id.replace('/', '_').lower()
            for existing_file in os.listdir("models"):
                if existing_file.lower().startswith(target_prefix) and "q4_k_m.gguf" in existing_file.lower():
                    is_cached = True
                    break
        else:
            try:
                from huggingface_hub import scan_cache_dir
                is_cached = any(repo.repo_id == repo_id for repo in scan_cache_dir().repos)
            except:
                pass
                
    if not is_cached:
        display_name = repo_id
        for name, repo in model_mapping.items():
            if repo == repo_id:
                display_name = name
                break
                
        st.warning(f"📥 Model **{display_name}** is not downloaded.")
        if st.button("🚀 Download Model", use_container_width=True):
            local_path = download_model_ui(repo_id)
            if local_path:
                st.rerun()
        st.stop()
        
        print(f"[SYSTEM] Beginning engine initialization...", flush=True)
    with st.status("🚀 Initializing Engine...", expanded=True) as status:
        st.write("🔄 Loading AI Models...")
        print(f"[SYSTEM] Loading model weights for {st.session_state['current_model']}...", flush=True)
        engine.load_models(quantization_mode=quant_mode)
        st.write(f"📂 Indexing Knowledge Base ({chunking_mode})...")
        print(f"[SYSTEM] Indexing knowledge base with {chunking_mode} chunking...", flush=True)
        engine.process_knowledge_base(chunking_mode=chunking_mode)
        status.update(label="✅ Engine Ready!", state="complete", expanded=False)
        print("[SYSTEM] Engine ready.", flush=True)
    st.session_state["models_loaded"] = True

# --- SESSION STATE FOR CHAT ---
if "messages" not in st.session_state:
    st.session_state["messages"] = db.load_messages("default_user")

# --- HEADER ---
st.title("🧠 AskBot: Advanced RAG Engine")
st.caption("v18 — Multi-Format Support | Hybrid Search | Quantization")

# --- CHAT DISPLAY ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Verified Citations"):
                for s in msg["sources"]:
                    # Handle both old (string) and new (dict) source formats
                    if isinstance(s, dict):
                        st.markdown(f"**[{s['id']}] {s['source']}**")
                        st.caption(s['text'])
                    else:
                        st.markdown(f"• {s}")

# --- CHAT INPUT ---
if prompt := st.chat_input("Ask about your knowledge base..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    db.save_message("default_user", "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.status("⚙️ Response Details", expanded= True) as status:
            # 1. Retrieval
            print(f"\n[SYSTEM] Received User Query: {prompt}", flush=True)
            st.write(f"Searching index {'(Hybrid+' if use_hybrid else '('}{'HyDE+' if use_hyde else ''}{'Rerank)' if use_rerank else ')'}...")
            print(f"[SYSTEM] Executing Retrieval Pipeline (Hybrid={use_hybrid}, HyDE={use_hyde}, Rerank={use_rerank})", flush=True)
            top_chunks, metrics = engine.retrieve(
                prompt, 
                k=3, 
                use_hybrid=use_hybrid, 
                use_hyde=use_hyde, 
                use_rerank=use_rerank, 
                use_parent=use_parent
            )
            
            if not top_chunks:
                response = "Not found."
                sources = []
                response_placeholder.markdown(response)
                gen_time = 0
            else:
                st.write(f"Found {len(top_chunks)} relevant segments.")
                context = ""
                sources_meta = []
                for i, c in enumerate(top_chunks):
                    # Use [Source 1], [Source 2] for LLM to cite
                    context += f"\n[Source {i+1}: {c['source']}]\n{c['text']}\n"
                    sources_meta.append({
                        "id": i+1, 
                        "source": c["source"], 
                        "text": c.get("retrieval_text", c["text"]) # Show exact snippet
                    })
                
                # 2. Generation
                st.write("Synthesizing answer...")
                print(f"[SYSTEM] Generation starting...", flush=True)
                full_response = ""
                
                streamer = engine.generate_stream(prompt, context, [
                    {"user": m["content"], "bot": st.session_state["messages"][i+1]["content"]} 
                    for i, m in enumerate(st.session_state["messages"][:-1]) if m["role"] == "user"
                ], api_key=st.session_state.get("google_api_key"))
                
                start_time = time.time()
                for new_text in streamer:
                    full_response += new_text
                    response_placeholder.markdown(full_response + "▌")
                
                gen_time = time.time() - start_time
                print(f"[SYSTEM] Generation finished in {gen_time:.2f}s", flush=True)
                response_placeholder.markdown(full_response)
                response = full_response
                
            # Show Telemetry inside the status block
            st.divider()
            col1, col2, col3 = st.columns(3)
            retrieval_time = metrics.get("semantic_time", 0) + metrics.get("keyword_time", 0) + metrics.get("hyde_gen_time", 0)
            col1.metric("Retrieval", f"{retrieval_time:.3f}s")
            col2.metric("Generation", f"{gen_time:.3f}s")
            col3.metric("Total", f"{retrieval_time + gen_time:.3f}s")
            
            # Show Chart inside the status block
            if top_chunks:
                chart_data = {
                    "Step": ["HyDE", "Semantic", "Keyword", "Fusion", "Rerank", "LLM Gen"],
                    "Time (s)": [
                        metrics.get("hyde_gen_time", 0),
                        metrics.get("semantic_time", 0), 
                        metrics.get("keyword_time", 0), 
                        metrics.get("fusion_time", 0), 
                        metrics.get("rerank_time", 0), 
                        gen_time
                    ]
                }
                st.bar_chart(chart_data, x="Step", y="Time (s)")

            status.update(label="⚙️ Response Details", state="complete", expanded=False)

        # Save to history
        st.session_state["messages"].append({
            "role": "assistant", 
            "content": response, 
            "sources": sources_meta,
            "metrics": metrics
        })
        db.save_message("default_user", "assistant", response, sources_meta, metrics)
        
        if sources_meta:
            with st.expander("📚 Verified Citations"):
                for s in sources_meta:
                    st.markdown(f"**[{s['id']}] {s['source']}**")
                    st.caption(s['text'])

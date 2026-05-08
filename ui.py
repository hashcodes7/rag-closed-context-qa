import streamlit as st
import time
import os
from core import RAGEngine, extract_text_from_file
import database as db

# Initialize database
db.init_db()

# =====================================================
# 🌊 RAGBOT V16 (Streamlit UI)
# =====================================================

st.set_page_config(page_title="SourceIQ RAG Engine", page_icon="🧠", layout="wide")

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stSidebar {
        background-color: rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
# --- INITIALIZE SESSION STATE ---
if "current_model" not in st.session_state:
    st.session_state["current_model"] = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"

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
    model_options = [
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
        "Custom Model..."
    ]
    
    selected_base = st.selectbox("Choose a model", model_options, 
                                 index=model_options.index(st.session_state["current_model"]) if st.session_state["current_model"] in model_options else 0)
    
    # 🆕 GGUF vs Standard Indicators
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
    uploaded_files = st.file_uploader("Upload Knowledge Files", type=["txt", "pdf", "docx"], accept_multiple_files=True)
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
        for f in os.listdir("knowledge_source"):
            ext = os.path.splitext(f)[1].lower()
            if ext in [".txt", ".pdf", ".docx"]:
                icon = "📄" if ext == ".txt" else "📕" if ext == ".pdf" else "📘"
                col_file, col_del = st.columns([0.8, 0.2])
                col_file.caption(f"{icon} {f}")
                if col_del.button("🗑️", key=f"del_{f}"):
                    os.remove(os.path.join("knowledge_source", f))
                    st.session_state["reindex_required"] = True
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

    st.info("SourceIQ V20 - Advanced Edition")

# --- INITIALIZE MODELS & DATA ---
if "reindex_required" not in st.session_state:
    st.session_state["reindex_required"] = False

if "models_loaded" not in st.session_state:
    with st.status("🚀 Initializing Engine...", expanded=True) as status:
        st.write("🔄 Loading AI Models...")
        engine.load_models(quantization_mode=quant_mode)
        st.write(f"📂 Indexing Knowledge Base ({chunking_mode})...")
        engine.process_knowledge_base(chunking_mode=chunking_mode)
        status.update(label="✅ Engine Ready!", state="complete", expanded=False)
    st.session_state["models_loaded"] = True

# --- RE-INDEX TRIGGER ---
if st.session_state["reindex_required"]:
    st.warning("⚠️ Knowledge base has changed. Re-index required to apply changes.")
    if st.button("🛠️ Re-index Knowledge Base Now"):
        with st.status("🏗️ Re-indexing...", expanded=True) as status:
            engine.process_knowledge_base(force_reindex=True, chunking_mode=chunking_mode)
            st.session_state["reindex_required"] = False
            status.update(label="✅ Re-indexed Successfully!", state="complete", expanded=False)
            st.rerun()

# --- SESSION STATE FOR CHAT ---
if "messages" not in st.session_state:
    st.session_state["messages"] = db.load_messages("default_user")

# --- HEADER ---
st.title("🧠 SourceIQ: Advanced RAG Engine")
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
            st.write(f"Searching index {'(Hybrid+' if use_hybrid else '('}{'HyDE+' if use_hyde else ''}{'Rerank)' if use_rerank else ')'}...")
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
                full_response = ""
                
                streamer = engine.generate_stream(prompt, context, [
                    {"user": m["content"], "bot": st.session_state["messages"][i+1]["content"]} 
                    for i, m in enumerate(st.session_state["messages"][:-1]) if m["role"] == "user"
                ])
                
                start_time = time.time()
                for new_text in streamer:
                    full_response += new_text
                    response_placeholder.markdown(full_response + "▌")
                
                gen_time = time.time() - start_time
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

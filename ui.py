import streamlit as st
import time
import os
from core import RAGEngine

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
if "current_model" not in st.session_state:
    st.session_state["current_model"] = "Qwen/Qwen2.5-0.5B-Instruct"

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
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "Custom Model..."
    ]
    
    selected_base = st.selectbox("Choose a model", model_options, 
                                 index=model_options.index(st.session_state["current_model"]) if st.session_state["current_model"] in model_options else 5)
    
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
    quant_mode = st.selectbox("Quantization Mode", ["4bit", "8bit", "full"], index=0)
    use_hybrid = st.toggle("Enable Hybrid Search (BM25)", value=True)
    
    st.divider()
    st.subheader("📂 Knowledge Manager")
    
    # File Uploader
    uploaded_files = st.file_uploader("Upload .txt files", type=["txt"], accept_multiple_files=True)
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
            if f.endswith(".txt"):
                col_file, col_del = st.columns([0.8, 0.2])
                col_file.caption(f"📄 {f}")
                if col_del.button("🗑️", key=f"del_{f}"):
                    os.remove(os.path.join("knowledge_source", f))
                    st.session_state["reindex_required"] = True
                    st.rerun()
    
    st.divider()
    if st.button("🔄 Force Engine Restart"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

    st.info("SourceIQ V17 - Interactive Edition")

# --- INITIALIZE MODELS & DATA ---
if "reindex_required" not in st.session_state:
    st.session_state["reindex_required"] = False

if "models_loaded" not in st.session_state:
    with st.status("🚀 Initializing Engine...", expanded=True) as status:
        st.write("🔄 Loading AI Models...")
        engine.load_models(quantization_mode=quant_mode)
        st.write("📂 Indexing Knowledge Base...")
        engine.process_knowledge_base()
        status.update(label="✅ Engine Ready!", state="complete", expanded=False)
    st.session_state["models_loaded"] = True

# --- RE-INDEX TRIGGER ---
if st.session_state["reindex_required"]:
    st.warning("⚠️ Knowledge base has changed. Re-index required to apply changes.")
    if st.button("🛠️ Re-index Knowledge Base Now"):
        with st.status("🏗️ Re-indexing...", expanded=True) as status:
            engine.process_knowledge_base(force_reindex=True)
            st.session_state["reindex_required"] = False
            status.update(label="✅ Re-indexed Successfully!", state="complete", expanded=False)
            st.rerun()

# --- SESSION STATE FOR CHAT ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- HEADER ---
st.title("🧠 SourceIQ: Advanced RAG Engine")
st.caption("v16 — FAISS HNSW | Hybrid Search | bitsandbytes Quantization")

# --- CHAT DISPLAY ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Full Sources"):
                selected_source = st.selectbox(f"Select a source to view ({msg['role']}_{st.session_state['messages'].index(msg)})", msg["sources"], key=f"src_{st.session_state['messages'].index(msg)}")
                if selected_source:
                    source_path = os.path.join("knowledge_source", selected_source)
                    if os.path.exists(source_path):
                        with open(source_path, "r", encoding="utf-8") as f:
                            st.text_area("Full Content", f.read(), height=200)
                    else:
                        st.error("Source file no longer exists.")

# --- CHAT INPUT ---
if prompt := st.chat_input("Ask about your knowledge base..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.status("🔍 Searching & Thinking...") as status:
            # 1. Retrieval
            st.write("Searching hybrid index...")
            top_chunks, metrics = engine.retrieve(prompt, k=3, use_hybrid=use_hybrid)
            
            if not top_chunks:
                response = "Not found."
                sources = []
            else:
                st.write(f"Found {len(top_chunks)} relevant segments.")
                context = ""
                sources = []
                for c in top_chunks:
                    context += f"\n[Source: {c['source']}]\n{c['text']}\n"
                    sources.append(c["source"])
                
                # 2. Generation
                st.write("Synthesizing answer...")
                response_placeholder = st.empty()
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
                
            status.update(label="✅ Response Generated", state="complete", expanded=False)

        # Show Telemetry
        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieval", f"{sum(metrics.values()):.3f}s")
        col2.metric("Generation", f"{gen_time:.3f}s")
        col3.metric("Total", f"{sum(metrics.values()) + gen_time:.3f}s")
        
        # Show Chart
        chart_data = {
            "Step": ["Semantic", "Keyword", "Fusion", "Rerank", "LLM Gen"],
            "Time (s)": [
                metrics["semantic_time"], 
                metrics["keyword_time"], 
                metrics["fusion_time"], 
                metrics["rerank_time"], 
                gen_time
            ]
        }
        st.bar_chart(chart_data, x="Step", y="Time (s)")

        # Save to history
        st.session_state["messages"].append({
            "role": "assistant", 
            "content": response, 
            "sources": list(set(sources))
        })
        
        if sources:
            with st.expander("📚 View Full Sources"):
                unique_srcs = list(set(sources))
                selected_source = st.selectbox("Select a source to view", unique_srcs, key=f"last_src_{len(st.session_state['messages'])}")
                if selected_source:
                    source_path = os.path.join("knowledge_source", selected_source)
                    if os.path.exists(source_path):
                        with open(source_path, "r", encoding="utf-8") as f:
                            st.text_area("Full Content", f.read(), height=200)
                    else:
                        st.error("Source file no longer exists.")

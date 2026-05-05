import streamlit as st
import time
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

# --- CACHED ENGINE INITIALIZATION ---
@st.cache_resource
def get_engine():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    return RAGEngine(model_name, embed_model_name, cross_encoder_model_name)

engine = get_engine()

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.title("⚙️ Engine Settings")
    st.divider()
    
    quant_mode = st.selectbox("Quantization Mode", ["4bit", "8bit", "full"], index=0)
    use_hybrid = st.toggle("Enable Hybrid Search (BM25)", value=True)
    
    st.divider()
    if st.button("🔄 Force Reload Knowledge Base"):
        st.cache_resource.clear()
        st.rerun()

    st.info("SourceIQ V16 - Streamlit Edition")

# --- INITIALIZE MODELS & DATA ---
if "models_loaded" not in st.session_state:
    with st.status("🚀 Initializing Engine...", expanded=True) as status:
        st.write("🔄 Loading AI Models...")
        engine.load_models(quantization_mode=quant_mode)
        st.write("📂 Indexing Knowledge Base...")
        engine.process_knowledge_base()
        status.update(label="✅ Engine Ready!", state="complete", expanded=False)
    st.session_state["models_loaded"] = True

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
        if "sources" in msg:
            with st.expander("📚 Sources"):
                st.write(", ".join(msg["sources"]))

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
            with st.expander("📚 Sources"):
                st.write(", ".join(list(set(sources))))

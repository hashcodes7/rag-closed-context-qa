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

# --- GLOBAL THEME INJECTOR ---
if st.session_state.get("theme_mode") == "light":
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }
    div[data-testid="stChatMessage"] {
        background-color: #e9ecef !important;
        color: #212529 !important;
        border-radius: 8px;
    }
    div[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #dee2e6;
    }
    .stMarkdown, p, span, label, h1, h2, h3, h4, h5, h6 {
        color: #212529 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- INITIALIZE SESSION STATE ---
if "current_model" not in st.session_state:
    st.session_state["current_model"] = "bartowski/Llama-3.2-1B-Instruct-GGUF"

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "user_role" not in st.session_state:
    st.session_state["user_role"] = None
if "uid" not in st.session_state:
    st.session_state["uid"] = None
if "auth_page" not in st.session_state:
    st.session_state["auth_page"] = "login"

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "chat"

if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"


if "use_hybrid" not in st.session_state:
    st.session_state["use_hybrid"] = True
if "use_hyde" not in st.session_state:
    st.session_state["use_hyde"] = False
if "use_rerank" not in st.session_state:
    st.session_state["use_rerank"] = True
if "use_parent" not in st.session_state:
    st.session_state["use_parent"] = True
if "quant_mode" not in st.session_state:
    st.session_state["quant_mode"] = "4bit"
if "chunking_mode" not in st.session_state:
    st.session_state["chunking_mode"] = "semantic"



# --- AUTO LOGIN VIA BROWSER QUERY PARAMS ---
if not st.session_state["logged_in"]:
    q_uid = None
    q_email = None
    try:
        if "uid" in st.query_params and "email" in st.query_params:
            q_uid = st.query_params["uid"]
            q_email = st.query_params["email"]
    except Exception:
        try:
            params = st.experimental_get_query_params()
            if "uid" in params and "email" in params:
                q_uid = params["uid"][0]
                q_email = params["email"][0]
        except Exception:
            pass
            
    if q_uid and q_email:
        user = db.get_user_by_credentials(q_uid, q_email)
        if user:
            st.session_state["logged_in"] = True
            st.session_state["uid"] = user["id"]
            st.session_state["username"] = user["username"]
            st.session_state["user_email"] = user["email"]
            st.session_state["user_role"] = user["role"]
            st.session_state["theme_mode"] = user.get("theme_preference", "dark")


# --- LOGIN / SIGNUP SCREENS ---
if not st.session_state["logged_in"]:
    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        if st.session_state["auth_page"] == "login":
            st.markdown("<h2 style='text-align: center;'>🧠 AskBot Sign In</h2>", unsafe_allow_html=True)
            st.caption("Access the secure local corporate RAG assistant.")
            
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Corporate Email ID", placeholder="name@freseniusmedicalcare.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Sign In", use_container_width=True)
                
                if submitted:
                    if not email.strip() or not password.strip():
                        st.error("Please enter both email and password.")
                    else:
                        user = db.authenticate_user(email, password)
                        if user:
                            st.session_state["logged_in"] = True
                            st.session_state["uid"] = user["id"]
                            st.session_state["username"] = user["username"]
                            st.session_state["user_email"] = user["email"]
                            st.session_state["user_role"] = user["role"]
                            st.session_state["theme_mode"] = user.get("theme_preference", "dark")

                            
                            # Save to query params for browser persistence
                            try:
                                st.query_params["uid"] = str(user["id"])
                                st.query_params["email"] = user["email"]
                            except Exception:
                                try:
                                    st.experimental_set_query_params(uid=str(user["id"]), email=user["email"])
                                except Exception:
                                    pass
                            st.rerun()
                        else:
                            st.error("Invalid email ID or password.")
            
            if st.button("New to AskBot? Sign Up here", use_container_width=True):
                st.session_state["auth_page"] = "signup"
                st.rerun()
                
        elif st.session_state["auth_page"] == "signup":
            st.markdown("<h2 style='text-align: center;'>📝 AskBot Sign Up</h2>", unsafe_allow_html=True)
            st.caption("Register below to access your isolated QA history.")
            
            with st.form("signup_form", clear_on_submit=False):
                username = st.text_input("Preferred Username", placeholder="e.g. harsh_verma")
                email = st.text_input("Corporate Email ID", placeholder="harsh.verma@freseniusmedicalcare.com")
                password = st.text_input("Password", type="password", placeholder="Password")
                confirm_password = st.text_input("Retype Password", type="password", placeholder="Retype Password")
                submitted = st.form_submit_button("Create Account", use_container_width=True)
                
                if submitted:
                    if not username.strip() or not email.strip() or not password.strip():
                        st.error("All fields are required.")
                    elif password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        success, detail = db.create_user(username, email, password)
                        if success:
                            st.session_state["auth_page"] = "signup_success"
                            st.rerun()
                        else:
                            st.error(detail)
            
            if st.button("Already have an account? Log In", use_container_width=True):
                st.session_state["auth_page"] = "login"
                st.rerun()
                
        elif st.session_state["auth_page"] == "signup_success":
            st.success("🎉 Signup Successful!")
            st.markdown("""
            Your corporate RAG account has been registered successfully.
            
            You can now log in using your email ID and password.
            """)
            if st.button("Proceed to Log In", use_container_width=True):
                st.session_state["auth_page"] = "login"
                st.rerun()
                
    st.stop()  # Stop execution of the rest of the application until logged in!

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
    # --- User Info & Logout ---
    st.markdown(f"👤 **Signed in as:** `{st.session_state['username']}`")
    st.caption(f"Role: `{st.session_state['user_role']}`")
    st.caption(f"Email: `{st.session_state['user_email']}`")
    
    if st.button("🚪 Log Out", use_container_width=True):
        # Clear query params for browser persistence
        try:
            st.query_params.clear()
        except Exception:
            try:
                st.experimental_set_query_params()
            except Exception:
                pass
        
        # Reset session states
        st.session_state["logged_in"] = False
        st.session_state["uid"] = None
        st.session_state["username"] = None
        st.session_state["user_email"] = None
        st.session_state["user_role"] = None
        st.session_state["auth_page"] = "login"
        st.session_state.pop("messages", None)
        st.rerun()


    if st.session_state.get("user_role") == "admin":
        if st.button("📂 Manage Knowledge Base", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "manage_kb"
            st.rerun()
            
        if st.button("👥 Manage Users", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "manage_users"
            st.rerun()
            
        if st.button("⚙️ Settings", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "settings"
            st.rerun()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        db.clear_history(username)
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    st.subheader("🎨 Theme Customization")
    theme_mode = st.radio("App Theme Mode", ["Dark Mode 🌙", "Light Mode ☀️"], 
                          index=0 if st.session_state["theme_mode"] == "dark" else 1,
                          key="sidebar_theme_radio")
    new_theme = "dark" if "Dark" in theme_mode else "light"
    if new_theme != st.session_state["theme_mode"]:
        st.session_state["theme_mode"] = new_theme
        db.update_user_theme(st.session_state["uid"], new_theme)
        st.rerun()

    st.divider()
    st.info("AskBot - Advanced Edition")


# --- INITIALIZE MODELS & DATA ---
if "reindex_required" not in st.session_state:
    st.session_state["reindex_required"] = False

# Extract initial variables from session state
quant_mode = st.session_state["quant_mode"]
chunking_mode = st.session_state["chunking_mode"]


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

username = st.session_state["username"]


# --- DEDICATED KNOWLEDGE BASE MANAGER ROUTING ---
if st.session_state.get("current_page") == "manage_kb":
    if st.session_state.get("user_role") != "admin":
        st.error("Access Denied: Only administrators can access this page.")
        st.session_state["current_page"] = "chat"
        st.rerun()
        
    st.title("📂 Document & Knowledge Base Manager")
    st.caption("View, search, upload, and delete documents supporting the AskBot RAG engine.")
    
    if st.button("⬅️ Back to Chat", use_container_width=True, type="secondary"):
        st.session_state["current_page"] = "chat"
        st.rerun()
        
    st.divider()
    
    # 📊 Telemetry Cards
    kb_folder = "knowledge_source"
    supported_exts = {".txt", ".pdf", ".docx", ".html", ".htm"}
    
    file_count = 0
    if os.path.exists(kb_folder):
        file_count = len([f for f in os.listdir(kb_folder) if os.path.splitext(f)[1].lower() in supported_exts])
        
    cache_exists = os.path.exists("vector_cache.pt")
    index_exists = os.path.exists("faiss_index.bin")
    chunks_loaded = getattr(engine, "chunks", None) and len(engine.chunks) > 0
    total_chunks = len(engine.chunks) if chunks_loaded else 0
    
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("📂 Total Documents", f"{file_count} files")
    col_t2.metric("⚡ Vector Cache Status", "Active (HNSW)" if cache_exists and index_exists else "Missing/Inactive")
    col_t3.metric("🧩 Indexed Chunks", f"{total_chunks} segments")
    
    st.divider()
    
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        st.subheader("📥 Ingest New Documents")
        uploaded_files = st.file_uploader("Upload Corporate Knowledge Files", type=["txt", "pdf", "docx", "html"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_path = os.path.join(kb_folder, uploaded_file.name)
                os.makedirs(kb_folder, exist_ok=True)
                if not os.path.exists(save_path):
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Successfully uploaded: {uploaded_file.name}")
                    st.session_state["reindex_required"] = True
                    st.rerun()
                    
        if st.session_state.get("reindex_required"):
            st.warning("⚠️ Files changed. Re-index recommended to rebuild HNSW Vector Index and BM25 index.")
            
        if st.button("🏗️ Force Re-Index Knowledge Base", use_container_width=True, type="primary"):
            with st.status("🏗️ Processing and Indexing Knowledge Base...", expanded=True) as status:
                engine.process_knowledge_base(force_reindex=True, chunking_mode=chunking_mode)
                st.session_state["reindex_required"] = False
                status.update(label="✅ Re-indexed Successfully!", state="complete", expanded=False)
                st.rerun()
                
    with col_right:
        st.subheader("🔍 Knowledge Base Directory")
        search_query = st.text_input("Search directory files...", placeholder="Type to filter file names...")
        
        if os.path.exists(kb_folder):
            try:
                indexed_sources = set([c["source"] for c in engine.chunks]) if getattr(engine, "chunks", None) else set()
            except Exception:
                indexed_sources = set()
                
            files_to_show = []
            for root, dirs, files in os.walk(kb_folder):
                for f in files:
                    abs_path = os.path.join(root, f)
                    relpath = os.path.relpath(abs_path, kb_folder).replace('\\', '/')
                    ext = os.path.splitext(f)[1].lower()
                    
                    if search_query.strip().lower() and search_query.strip().lower() not in relpath.lower():
                        continue
                        
                    files_to_show.append((abs_path, relpath, ext))
                    
            if not files_to_show:
                st.info("No matching knowledge source files found.")
            else:
                for abs_path, relpath, ext in files_to_show:
                    parts = relpath.split('/')
                    namespace = parts[0] if len(parts) > 1 else (parts[0] if parts else 'root')
                    
                    if ext not in supported_exts:
                        status_char = "✗"
                        status_title = "Unsupported file type"
                    elif relpath in indexed_sources:
                        status_char = "✓"
                        status_title = "Indexed"
                    else:
                        status_char = "○"
                        status_title = "Supported (not indexed)"
                        
                    icon = "📄" if ext == ".txt" else "📕" if ext == ".pdf" else "📘" if ext == ".docx" else "🌐" if ext in {".html", ".htm"} else "📁"
                    
                    col_file_info, col_file_del = st.columns([0.85, 0.15])
                    with col_file_info:
                        st.markdown(f"**{status_char} {icon} {relpath}**  *(Namespace: `{namespace}`)*")
                        st.caption(f"Status: {status_title}")
                    with col_file_del:
                        del_key = f"del_mgr_{relpath.replace('/', '_')}"
                        if st.button("🗑️", key=del_key, help=f"Delete {relpath}"):
                            os.remove(abs_path)
                            st.session_state["reindex_required"] = True
                            st.rerun()
    st.stop()

# --- DEDICATED USER MANAGEMENT ROUTING (ADMIN ONLY) ---
if st.session_state.get("current_page") == "manage_users":
    if st.session_state.get("user_role") != "admin":
        st.error("Access Denied: You do not have permission to view this page.")
        st.session_state["current_page"] = "chat"
        st.rerun()
        
    st.title("👥 User Administration Dashboard")
    st.caption("Admin Mode — View user accounts, change roles, edit details, and delete profiles with history.")
    
    if st.button("⬅️ Back to Chat", use_container_width=True, type="secondary"):
        st.session_state["current_page"] = "chat"
        st.rerun()
        
    st.divider()
    
    # 🔍 Search & Directory
    all_users = db.get_all_users()
    
    search_query = st.text_input("🔍 Search users by name or email...", placeholder="Type to filter...")
    
    filtered_users = []
    for u in all_users:
        if search_query.strip().lower():
            if (search_query.strip().lower() not in u["username"].lower()) and (search_query.strip().lower() not in u["email"].lower()):
                continue
        filtered_users.append(u)
        
    if not filtered_users:
        st.info("No matching corporate users found.")
    else:
        st.write(f"Showing {len(filtered_users)} registered users:")
        st.divider()
        
        for u in filtered_users:
            u_id = u["id"]
            u_username = u["username"]
            u_email = u["email"]
            u_role = u["role"]
            u_created = u["created_at"]
            
            # Use unique keys for each edit form
            form_key = f"user_edit_form_{u_id}"
            
            # Display user details in an expander for clean visual separation
            with st.expander(f"👤 {u_username} — {u_email} ({u_role})"):
                with st.form(form_key, clear_on_submit=False):
                    col_u1, col_u2 = st.columns(2)
                    with col_u1:
                        # Email acts as the immutable corporate identifier (immutable/read-only PK)
                        st.text_input("Email ID (Corporate Login ID — Locked)", value=u_email, disabled=True)
                        edit_username = st.text_input("Preferred Username", value=u_username)
                    with col_u2:
                        role_options = ["user", "admin"]
                        edit_role = st.selectbox("Assign Privilege Role", role_options, index=role_options.index(u_role))
                        st.text_input("Registered Since", value=u_created, disabled=True)
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        save_submitted = st.form_submit_button("💾 Save User Changes", use_container_width=True)
                        if save_submitted:
                            if not edit_username.strip():
                                st.error("Username cannot be empty.")
                            else:
                                success, msg = db.update_user(u_id, edit_username, edit_role)
                                if success:
                                    st.success(msg)
                                    # If the edited user is the active logged-in user, refresh their session
                                    if u_id == st.session_state.get("uid"):
                                        st.session_state["username"] = edit_username.strip()
                                        st.session_state["user_role"] = edit_role
                                    st.rerun()
                                else:
                                    st.error(msg)
                                    
                    with col_btn2:
                        # Safety lock to prevent self-deletion or lockout
                        is_self = (u_id == st.session_state.get("uid"))
                        delete_disabled = is_self
                        
                        delete_btn = st.form_submit_button("🗑️ Delete Account & History", use_container_width=True, type="primary", disabled=delete_disabled)
                        if delete_btn:
                            success = db.delete_user_and_history(u_id, u_username)
                            if success:
                                st.success(f"User {u_username} and their chat context were successfully purged.")
                                st.rerun()
                            else:
                                st.error("Failed to delete user.")
                                
                        if is_self:
                            st.caption("🔒 *Self-deletion is disabled to prevent lockout.*")
    st.stop()

# --- DEDICATED SETTINGS ROUTING ---
if st.session_state.get("current_page") == "settings":
    if st.session_state.get("user_role") != "admin":
        st.error("Access Denied: Only administrators can access this page.")
        st.session_state["current_page"] = "chat"
        st.rerun()
        
    st.title("⚙️ RAG Engine Configurations")
    st.caption("Configure local models, retrieval algorithms, quantization modes, and system restarts.")
    
    if st.button("⬅️ Back to Chat", use_container_width=True, type="secondary"):
        st.session_state["current_page"] = "chat"
        st.rerun()
        
    st.divider()
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader("🤖 Model Selection")
        
        model_mapping = {
            "Meta Llama 3.2 1B Instruct GGUF": "bartowski/Llama-3.2-1B-Instruct-GGUF",
            "Google Gemini 1.5 Flash": "gemini-1.5-flash",
            "Google Gemini 1.5 Pro": "gemini-1.5-pro",
            "Microsoft Phi-3.5 Mini Instruct GGUF": "bartowski/Phi-3.5-mini-instruct-GGUF",
            "Microsoft Phi-2 GGUF": "TheBloke/phi-2-GGUF",
            "Custom Model...": "Custom Model..."
        }
        model_options = list(model_mapping.values())
        display_options = list(model_mapping.keys())
        
        # Find display name for the current model in session state
        current_display = "Meta Llama 3.2 1B Instruct GGUF"
        for disp, raw in model_mapping.items():
            if raw == st.session_state.get("current_model"):
                current_display = disp
                break
                
        if current_display == "Meta Llama 3.2 1B Instruct GGUF" and st.session_state.get("current_model") and st.session_state["current_model"] not in model_mapping.values():
            current_display = "Custom Model..."
            
        selected_display = st.selectbox("Choose a model", display_options, 
                                     index=display_options.index(current_display))
        selected_base = model_mapping.get(selected_display, selected_display)
        
        # GGUF vs API vs Standard Indicators
        if selected_base.startswith("gemini-"):
            st.success("☁️ **Google Gemini API**")
            st.caption("Fast and powerful cloud inference.")
            api_key = st.text_input("Google API Key", type="password", value=st.session_state.get("google_api_key", ""))
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
                



                
    with col_r:
        st.subheader("🛠️ Algorithm & Engine Control")
        
        st.session_state["use_hybrid"] = st.toggle("Hybrid Search (BM25)", value=st.session_state["use_hybrid"], help="Combines keyword search with vector search.")
        st.session_state["use_hyde"] = st.toggle("HyDE Expansion", value=st.session_state["use_hyde"], help="Generates an ideal answer first to improve retrieval.")
        st.session_state["use_rerank"] = st.toggle("Cross-Encoder Rerank", value=st.session_state["use_rerank"], help="Uses a secondary model to refine result relevance.")
        st.session_state["use_parent"] = st.toggle("Parent-Doc Context", value=st.session_state["use_parent"], help="Retrieves the full paragraph context for the LLM.")
        
        st.divider()
        
        quant_options = ["4bit", "8bit", "full"]
        st.session_state["quant_mode"] = st.selectbox("Quantization Mode", quant_options, index=quant_options.index(st.session_state["quant_mode"]))
        
        chunk_options = ["semantic", "recursive"]
        st.session_state["chunking_mode"] = st.radio("Chunking Strategy", chunk_options, index=chunk_options.index(st.session_state["chunking_mode"]))
        
        st.divider()
        
        if st.button("🔄 Force Engine Restart", use_container_width=True, type="primary"):
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()
                
    st.stop()


# --- SESSION STATE FOR CHAT ---
if "messages" not in st.session_state:
    st.session_state["messages"] = db.load_messages(username)

# Extract config variables from session state
use_hybrid = st.session_state["use_hybrid"]
use_hyde = st.session_state["use_hyde"]
use_rerank = st.session_state["use_rerank"]
use_parent = st.session_state["use_parent"]
quant_mode = st.session_state["quant_mode"]
chunking_mode = st.session_state["chunking_mode"]



# --- HEADER ---
st.title("🧠 AskBot: Advanced RAG Engine")
st.caption("v18 — Multi-Format Support | Hybrid Search | Quantization")

# --- CHAT DISPLAY ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f"**Prompt ({username})**")
        else:
            st.markdown(f"**Response ({username})**")
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
    db.save_message(username, "user", prompt)
    with st.chat_message("user"):
        st.markdown(f"**Prompt ({username})**")
        st.markdown(prompt)


    # Generate response
    with st.chat_message("assistant"):
        st.markdown(f"**Response ({username})**")
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
                
                # Construct chat history securely by pairing user prompts with their assistant responses
                chat_history = []
                all_messages = st.session_state["messages"][:-1]
                for idx, m in enumerate(all_messages):
                    if m["role"] == "user":
                        bot_content = ""
                        if idx + 1 < len(all_messages) and all_messages[idx + 1]["role"] == "assistant":
                            bot_content = all_messages[idx + 1]["content"]
                        chat_history.append({"user": m["content"], "bot": bot_content})

                streamer = engine.generate_stream(
                    prompt, 
                    context, 
                    chat_history, 
                    api_key=st.session_state.get("google_api_key")
                )

                
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
        db.save_message(username, "assistant", response, sources_meta, metrics)
        
        if sources_meta:
            with st.expander("📚 Verified Citations"):
                for s in sources_meta:
                    st.markdown(f"**[{s['id']}] {s['source']}**")
                    st.caption(s['text'])

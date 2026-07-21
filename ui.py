import streamlit as st
import time
import os
from core import RAGEngine, extract_text_from_file
import database as db

# Initialize database
db.init_db()

# Fallback phrase the model is instructed to emit when the context is unrelated.
# Sometimes the model emits this phrase AND then continues with an answer anyway.
# In that case we strip the fallback line so only the real answer is shown.
FALLBACK_PHRASE = "I think this info isn't yet added to my knowledge base."

def strip_fallback_prefix(text):
    stripped = text.lstrip()
    if stripped.startswith(FALLBACK_PHRASE):
        remainder = stripped[len(FALLBACK_PHRASE):].lstrip(" .\n\r\t")
        if remainder:
            return remainder
    return text

# =====================================================
# 🎯 APPLICATION SCOPING (Part 1: application-specific RAG)
# -----------------------------------------------------
# The user picks which application their questions are about. The selection
# prepends an application prompt to the system prompt so that ambiguous
# questions (e.g. "how is a user created") are answered strictly in the
# context of the chosen application instead of leaking another app's details.
# =====================================================
DEFAULT_APP = "All Applications"

APP_DESCRIPTIONS = {
    "Trackwise": "TrackWise is a Quality Management System (QMS) used to manage quality processes such as CAPA, deviations, complaints, audits, change control, and quality records.",
    "ThingWorx": "ThingWorx is PTC's Industrial IoT (IIoT) platform used to build and run connected applications, model Things/data, and create dashboards (mashups).",
    "Polarion": "Polarion is Siemens' Application Lifecycle Management (ALM) tool used for requirements management, test management, and work-item tracking.",
    "Windchill GPDM": "Windchill GPDM (Global Product Data Management) is PTC's PLM system used to manage product data, parts, documents, BOMs, and change management.",
}

# Order shown in the dropdown; DEFAULT_APP first so behaviour is unchanged until a choice is made.
APP_OPTIONS = [DEFAULT_APP] + list(APP_DESCRIPTIONS.keys())

# Maps each app to the knowledge_source top-level folder(s) that become the chunk
# "namespace" during indexing. Used to filter retrieval so only that app's documents
# are searched. Note: the Windchill folder on disk is "Windchill GPDM & Windchill NA".
APP_NAMESPACES = {
    "Trackwise": ["Trackwise"],
    "ThingWorx": ["ThingWorx"],
    "Polarion": ["Polarion"],
    "Windchill GPDM": ["Windchill GPDM & Windchill NA"],
}


def get_allowed_namespaces(app):
    """Return the list of allowed chunk namespaces for an app, or None for the default (no filter)."""
    if not app or app == DEFAULT_APP:
        return None
    return APP_NAMESPACES.get(app)


def build_app_prompt(app):
    """Return the application-scoping system-prompt block, or None for the default."""
    if not app or app == DEFAULT_APP:
        return None
    description = APP_DESCRIPTIONS.get(app, "")
    return (
        "APPLICATION CONTEXT (HIGHEST PRIORITY SCOPE):\n"
        f"- The user is asking specifically about the \"{app}\" application. {description}\n"
        f"- Treat every question as being about {app}, even when the question does not name any application.\n"
        f"- Many support questions are ambiguous and could apply to several systems (for example: "
        f"'how is a user created', 'how to reset a password', 'how to configure roles', 'how to import data'). "
        f"Always interpret and answer such questions strictly in the context of {app}.\n"
        f"- When using the retrieved context below, prefer information that pertains to {app}. "
        f"If a retrieved passage clearly belongs to a different application, do not present it as if it applies to {app}.\n"
        f"- If the context only contains information about a different application and nothing about {app}, "
        f"use the standard fallback phrase instead of answering with the wrong application's details."
    )

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
# 🌊 CognIQ (Streamlit UI)
# =====================================================

st.set_page_config(page_title="CognIQ", page_icon="🧠", layout="wide")

# Inject Custom CSS to match the mockup exactly
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background-color: #ffffff !important;
}

.block-container {
    padding-bottom: 3.5rem !important;
}

section[data-testid="stSidebar"] {
    background-color: #f4f3ef !important;
    border-right: 1px solid #e2e8f0 !important;
}

div[data-testid="stChatMessageAvatar"] {
    display: none !important;
}
div[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 10px 0 !important;
}

.user-msg-container {
    display: flex;
    justify-content: flex-end;
    width: 100%;
    margin: 8px 0;
}
.user-msg-bubble {
    background-color: #f7f7f5 !important;
    color: #111827 !important;
    border-radius: 18px !important;
    padding: 12px 18px !important;
    max-width: 70% !important;
    font-size: 15px !important;
    font-weight: 400;
    line-height: 1.5;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}

.assistant-container {
    width: 100%;
    margin: 8px 0;
}
.assistant-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
    font-size: 14px;
}
.assistant-logo {
    font-size: 16px;
}
.assistant-name {
    font-weight: 600;
    color: #111827;
}
.assistant-tag {
    color: #6b7280;
}
.assistant-card {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02) !important;
    font-size: 15px;
    line-height: 1.6;
    color: #1f2937;
}
.assistant-body {
    white-space: pre-line;
}
.card-divider {
    border: 0;
    border-top: 1px solid #e2e8f0;
    margin: 16px 0;
}
.sources-container {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.sources-title {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #9ca3af;
}
.source-chip {
    display: flex;
    align-items: center;
    gap: 10px;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 13px;
}
.badge {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
    text-transform: uppercase;
}
.badge-ticket {
    background-color: #dbeafe !important;
    color: #2563eb !important;
}
.badge-kb {
    background-color: #dcfce7 !important;
    color: #166534 !important;
}
.source-text {
    flex-grow: 1;
    color: #374151;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.match-pct {
    color: #9ca3af;
    font-size: 12px;
    white-space: nowrap;
}
.utility-row {
    display: flex;
    gap: 16px;
    margin-top: 8px;
    padding-left: 4px;
    font-size: 13px;
    color: #9ca3af;
}
.utility-item {
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 4px;
    transition: color 0.2s;
}
.utility-item:hover {
    color: #2563eb;
}


/* Hide default streamlit UI headers and menus completely to reclaim space */
#MainMenu {display: none !important;}
header {display: none !important;}
footer {display: none !important;}

/* Adjust chat input container positioning and layout */
div[data-testid="stChatInput"] {
    bottom: 16px !important;
    padding-bottom: 0px !important;
    padding-top: 0px !important;
    margin: 0px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    background-color: transparent !important;
}

/* Ensure inner form elements have no bottom margins */
div[data-testid="stChatInput"] form {
    margin: 0px !important;
    padding: 0px !important;
    position: relative !important;
}

/* Add top padding to textarea to make height a bit larger while keeping writing area clean */
div[data-testid="stChatInput"] textarea {
    padding-top: 42px !important;
    min-height: 95px !important;
}

/* Position app pill floating inside top-left of the chatbox */
.st-key-selected_app {
    position: absolute !important;
    top: 8px !important;
    left: 14px !important;
    z-index: 99 !important;
    width: auto !important;
    max-width: 220px !important;
}

/* Hide default stacked label */
.st-key-selected_app label { display: none !important; }

/* Turn BaseWeb select control into a rounded compact pill */
.st-key-selected_app div[data-baseweb="select"] > div {
    border-radius: 9999px !important;
    background-color: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    min-height: 28px !important;
    height: 28px !important;
    padding-left: 10px !important;
    padding-right: 6px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.15s ease !important;
}
.st-key-selected_app div[data-baseweb="select"] > div:hover {
    border-color: #94a3b8 !important;
    background-color: #e2e8f0 !important;
}
.st-key-selected_app div[data-baseweb="select"] div,
.st-key-selected_app div[data-baseweb="select"] span {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #374151 !important;
}

/* Add tagline above the chat input box */
div[data-testid="stChatInput"]::before {
    content: "Grounded in your tickets & KB · every answer cites its source";
    display: block;
    text-align: center;
    font-size: 12px;
    color: #9ca3af;
    margin-bottom: 6px;
    width: 100%;
}

/* Custom styles for sidebar elements */
.sidebar-branding {
    margin-bottom: 20px;
    margin-top: 10px;
}
.sidebar-title {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
    margin: 0;
}
.sidebar-subtitle {
    font-size: 13px;
    color: #6b7280;
    margin: 0;
}

/* Custom New Chat Button Styling */
div.stButton > button {
    background-color: #dbeafe !important;
    color: #2563eb !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 10px 16px !important;
    transition: all 0.2s !important;
}
div.stButton > button:hover {
    background-color: #bfdbfe !important;
    transform: translateY(-1px);
}

/* KB Status Card */
.kb-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px;
    margin-top: 16px;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
}
.kb-card-title {
    font-size: 14px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 4px;
}
.kb-card-status {
    font-size: 12px;
    color: #6b7280;
    display: flex;
    align-items: center;
    gap: 6px;
}
.status-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
}

/* Fixed User Profile Card at Sidebar Bottom */
div[data-testid="stSidebarUserContent"] {
    display: flex;
    flex-direction: column;
    height: 100vh;
}
.profile-container {
    margin-top: auto !important;
    margin-bottom: 20px !important;
    display: flex;
    align-items: center;
    gap: 12px;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 10px 14px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
}
.profile-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background-color: #dbeafe;
    color: #2563eb;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 16px;
}
.profile-info {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
}
.profile-name {
    font-size: 14px;
    font-weight: 600;
    color: #111827;
}
.profile-role {
    font-size: 12px;
    color: #6b7280;
}
.profile-chevron {
    color: #9ca3af;
    font-size: 14px;
}

/* Custom Header layout */
.main-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 16px;
}
.engine-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    color: #374151;
    font-weight: 500;
}
.engine-status .status-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
}
.model-badge {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 13px;
    color: #4b5563;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* .input-tagline CSS removed - handled via ::before pseudo-element */
</style>
""", unsafe_allow_html=True)

# Helper function to format citations as custom HTML chips matching the mockup
def format_sources_html(sources):
    if not sources:
        return ""
        
    import os
    chips = []
    for idx, s in enumerate(sources):
        meta = {}
        if isinstance(s, dict):
            src_name = s.get("source", "")
            match_score = s.get("score")
            if match_score is None:
                match_score = 0.95 - (idx * 0.05)
            match_pct = int(match_score * 100) if match_score <= 1.0 else int(match_score)
            meta = s.get("metadata", {}) or {}
        else:
            src_name = str(s)
            match_pct = 90 - (idx * 5)
            
        is_ticket = src_name.lower().endswith((".xlsx", ".xls")) or "inc" in src_name.lower() or "ticket" in src_name.lower()
        badge_class = "badge-ticket" if is_ticket else "badge-kb"
        badge_text = "Ticket" if is_ticket else "KB"
        
        basename = os.path.basename(src_name)
        if len(basename) > 50:
            basename = basename[:47] + "..."
            
        tooltip_parts = [f"Source: {src_name}"]
        meta_info = ""
        if meta:
            if meta.get("title"):
                tooltip_parts.append(f"Title: {meta['title']}")
            if meta.get("author"):
                tooltip_parts.append(f"Author: {meta['author']}")
            if meta.get("creator"):
                tooltip_parts.append(f"Creator/Editor: {meta['creator']}")
            if meta.get("created_time"):
                tooltip_parts.append(f"Created: {meta['created_time']}")
            if meta.get("modified_time"):
                tooltip_parts.append(f"Modified: {meta['modified_time']}")
            if meta.get("file_size_bytes"):
                tooltip_parts.append(f"Size: {meta['file_size_bytes']/1024:.1f} KB")
                
            author_info = meta.get("author") or meta.get("creator") or ""
            date_info = meta.get("modified_time") or meta.get("created_time") or ""
            if date_info:
                date_info = date_info.split()[0]
                
            if author_info and date_info:
                meta_info = f" ({author_info} · {date_info})"
            elif author_info:
                meta_info = f" ({author_info})"
            elif date_info:
                meta_info = f" ({date_info})"
                
        tooltip = " | ".join(tooltip_parts)
        chip_html = f'<div class="source-chip"><span class="badge {badge_class}">{badge_text}</span><span class="source-text" title="{tooltip}">{basename}{meta_info}</span><span class="match-pct">{match_pct}% match</span></div>'
        chips.append(chip_html)
        
    if not chips:
        return ""
        
    return f'<hr class="card-divider"/><div class="sources-container"><div class="sources-title">Sources</div>{"".join(chips)}</div>'




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

if "selected_app" not in st.session_state:
    st.session_state["selected_app"] = DEFAULT_APP

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
            st.session_state["theme_mode"] = "dark"


# --- LOGIN / SIGNUP SCREENS ---
if not st.session_state["logged_in"]:
    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        if st.session_state["auth_page"] == "login":
            st.image("media/combined_logo.png", use_container_width=True)
            st.markdown("<h2 style='text-align: center; margin-top: 10px;'>CognIQ</h2>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center; color: #a0aec0; font-weight: 500; margin-top: -5px;'>Ask once. Resolve faster.</h4>", unsafe_allow_html=True)
            st.caption("AMS support intelligence — answers from past ticket resolutions & knowledge base articles")
            
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
                            st.session_state["theme_mode"] = "dark"

                            
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
            
            if st.button("New to CognIQ? Sign Up here", use_container_width=True):
                st.session_state["auth_page"] = "signup"
                st.rerun()
                
        elif st.session_state["auth_page"] == "signup":
            st.image("media/combined_logo.png", use_container_width=True)
            st.markdown("<h2 style='text-align: center; margin-top: 10px;'>CognIQ Sign Up</h2>", unsafe_allow_html=True)
            st.caption("Register below to access your isolated QA history.")
            
            with st.form("signup_form", clear_on_submit=False):
                username = st.text_input("Preferred Username", placeholder="e.g. user_name")
                email = st.text_input("Corporate Email ID", placeholder="name@freseniusmedicalcare.com")
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
            st.image("media/combined_logo.png", use_container_width=True)
            st.success("🎉 Signup Successful!")
            st.markdown("""
            Your corporate RAG account has been registered successfully.
            
            You can now log in using your email ID and password.
            """)
            if st.button("Proceed to Log In", use_container_width=True):
                st.session_state["auth_page"] = "login"
                st.rerun()
                
    st.stop()  # Stop execution of the rest of the application until logged in!

username = st.session_state["username"]

# --- CACHED ENGINE INITIALIZATION ---
@st.cache_resource
def get_engine(model_name):
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    return RAGEngine(model_name, embed_model_name, cross_encoder_model_name)

engine = get_engine(st.session_state["current_model"])

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.image("media/combined_logo.png", use_container_width=True)
    st.markdown("""
    <div class="sidebar-branding">
        <div class="sidebar-title">CognIQ</div>
        <div class="sidebar-subtitle">advanced RAG engine</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("➕ New chat", use_container_width=True):
        db.clear_history(username)
        st.session_state["messages"] = []
        st.session_state["current_page"] = "chat"
        st.rerun()

    # Dynamic Knowledge Base Telemetry Card
    kb_folder = "knowledge_source"
    file_count = 0
    total_size_bytes = 0
    if os.path.exists(kb_folder):
        for root, dirs, files in os.walk(kb_folder):
            for f in files:
                file_count += 1
                total_size_bytes += os.path.getsize(os.path.join(root, f))
    total_size_kb = total_size_bytes / 1024
    if total_size_kb > 1024:
        kb_size_str = f"{total_size_kb/1024:.1f} MB"
    else:
        kb_size_str = f"{total_size_kb:.1f} KB"
        
    st.markdown(f"""
    <div class="kb-card">
        <div class="kb-card-title">📂 Knowledge base</div>
        <div class="kb-card-status">
            <span class="status-dot"></span> {file_count:,} documents · {kb_size_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Admin Control expander
    if st.session_state.get("user_role") == "admin":
        st.markdown("<br>", unsafe_allow_html=True)
        with st.sidebar.expander("⚙️ Admin Console"):
            if st.button("📂 Manage KB", use_container_width=True):
                st.session_state["current_page"] = "manage_kb"
                st.rerun()
            if st.button("👥 Manage Users", use_container_width=True):
                st.session_state["current_page"] = "manage_users"
                st.rerun()
            if st.button("🔧 Settings", use_container_width=True):
                st.session_state["current_page"] = "settings"
                st.rerun()

    if st.button("🚪 Log Out", use_container_width=True, type="secondary"):
        try:
            st.query_params.clear()
        except Exception:
            try:
                st.experimental_set_query_params()
            except Exception:
                pass
        st.session_state["logged_in"] = False
        st.session_state["uid"] = None
        st.session_state["username"] = None
        st.session_state["user_role"] = None
        st.session_state["auth_page"] = "login"
        st.session_state.pop("messages", None)
        st.rerun()

    # Fixed User Profile Card at Sidebar Bottom
    display_username = st.session_state.get('username', '')
    first_letter = display_username[0].upper() if display_username else 'C'
    st.markdown(f'<div class="profile-container"><div class="profile-avatar">{first_letter}</div><div class="profile-info"><div class="profile-name">{display_username}</div><div class="profile-role">{st.session_state.get("user_role", "").title()}</div></div><div class="profile-chevron">👤</div></div>', unsafe_allow_html=True)


# --- INITIALIZE MODELS & DATA ---
if "reindex_required" not in st.session_state:
    st.session_state["reindex_required"] = False

# Extract initial variables from session state
quant_mode = st.session_state["quant_mode"]
chunking_mode = st.session_state["chunking_mode"]


if "models_loaded" not in st.session_state:
    st.session_state["models_loaded"] = False
if "engine_error" not in st.session_state:
    st.session_state["engine_error"] = None

# Only initialize engine if on chat page and not already loaded/errored
if not st.session_state["models_loaded"] and st.session_state.get("current_page") == "chat" and not st.session_state.get("engine_error"):
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
    try:
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
        st.session_state["engine_error"] = None
    except Exception as e:
        import traceback
        err_trace = traceback.format_exc()
        print(f"[SYSTEM] Engine load failed:\n{err_trace}", flush=True)
        st.session_state["engine_error"] = f"{str(e)}"

# --- DEDICATED KNOWLEDGE BASE MANAGER ROUTING ---
if st.session_state.get("current_page") == "manage_kb":
    if st.session_state.get("user_role") != "admin":
        st.error("Access Denied: Only administrators can access this page.")
        st.session_state["current_page"] = "chat"
        st.rerun()
        
    st.image("media/combined_logo.png", width=340)
    st.title("📂 Document & Knowledge Base Manager")
    st.caption("View, search, upload, and delete documents supporting the CognIQ RAG engine.")
    
    if st.button("⬅️ Back to Chat", use_container_width=True, type="secondary"):
        st.session_state["current_page"] = "chat"
        st.rerun()
        
    st.divider()
    
    # 📊 Telemetry Cards
    kb_folder = "knowledge_source"
    supported_exts = {".txt", ".pdf", ".docx", ".html", ".htm", ".xlsx", ".xlsm"}
    
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
        
        # Select target application
        upload_app = st.selectbox("Select Target Application Namespace", ["Windchill GPDM", "Polarion", "ThingWorx", "Trackwise"])
        target_subfolder = ""
        if upload_app == "Windchill GPDM":
            target_subfolder = "Windchill GPDM & Windchill NA"
        elif upload_app == "Polarion":
            target_subfolder = "Polarion"
        elif upload_app == "ThingWorx":
            target_subfolder = "ThingWorx"
        elif upload_app == "Trackwise":
            target_subfolder = "Trackwise"
            
        target_dir = os.path.join(kb_folder, target_subfolder)
        
        uploaded_files = st.file_uploader(f"Upload Corporate Knowledge Files to {upload_app}", type=["txt", "pdf", "docx", "html", "xlsx", "xlsm"], accept_multiple_files=True)
        if uploaded_files:
            os.makedirs(target_dir, exist_ok=True)
            for uploaded_file in uploaded_files:
                save_path = os.path.join(target_dir, uploaded_file.name)
                if not os.path.exists(save_path):
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Successfully uploaded: {uploaded_file.name} to {upload_app}")
                    st.session_state["reindex_required"] = True
                    st.rerun()
                    
        if st.session_state.get("reindex_required"):
            st.warning("⚠️ Files changed. Re-indexing is recommended to sync changes.")
            
        col_idx1, col_idx2 = st.columns(2)
        with col_idx1:
            st.write("**Incremental Update**")
            st.caption("reindex only new files")
            if st.button("⚡ Incremental Update", use_container_width=True, type="secondary"):
                with st.status("⚡ Updating Knowledge Base incrementally...", expanded=True) as status:
                    engine.process_knowledge_base(force_reindex=False, incremental=True, chunking_mode=chunking_mode)
                    st.session_state["reindex_required"] = False
                    status.update(label="✅ Incremental Update Complete!", state="complete", expanded=False)
                    st.rerun()
                    
        with col_idx2:
            st.write("**Complete Re-Index**")
            st.caption("reindex every single file (this will take much more time)")
            if st.button("🏗️ Complete Re-Index", use_container_width=True, type="primary"):
                with st.status("🏗️ Rebuilding Knowledge Base completely...", expanded=True) as status:
                    engine.process_knowledge_base(force_reindex=True, incremental=False, chunking_mode=chunking_mode)
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
                        
                    icon = "📄" if ext == ".txt" else "📕" if ext == ".pdf" else "📘" if ext == ".docx" else "🌐" if ext in {".html", ".htm"} else "📊" if ext in {".xlsx", ".xlsm"} else "📁"
                    
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
        
    st.image("media/combined_logo.png", width=340)
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
        
    st.image("media/combined_logo.png", width=340)
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
                st.session_state.pop("engine_error", None)  # Reset error status
                st.rerun()
                
        if st.session_state.get("engine_error"):
            st.error(f"❌ RAG Engine Error: {st.session_state['engine_error']}")
            if st.button("🔄 Retry Engine Load", use_container_width=True):
                st.session_state.pop("models_loaded", None)
                st.session_state.pop("engine_error", None)
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



# --- HEADER & STATUS BANNERS ---
if st.session_state.get("engine_error"):
    st.error(f"⚠️ RAG Engine Offline: {st.session_state['engine_error']}")
    st.info("You can still use the admin dashboard pages. To retry, switch models or visit Settings.")

st.markdown("""
<div class="main-header-row">
    <div class="engine-status">
        <span class="status-dot"></span> Engine ready
    </div>
    <div class="model-badge">
        ⚙️ Cognizant in-house model
    </div>
</div>
""", unsafe_allow_html=True)

# Empty-chat landing page logo layout
if not st.session_state["messages"]:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
    with col_logo2:
        st.image("media/combined_logo.png", use_container_width=True)
        st.markdown("<h1 style='text-align: center; margin-top: 10px; color: #111827;'>🧠 CognIQ</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: #6b7280; font-weight: 500; margin-top: -5px;'>Ask once. Resolve faster.</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 14px;'>AMS support intelligence — answers from past ticket resolutions & knowledge base articles</p>", unsafe_allow_html=True)

# --- CHAT DISPLAY ---
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg-container"><div class="user-msg-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        sources_html = format_sources_html(msg.get("sources", []))
        st.markdown(f'<div class="assistant-container"><div class="assistant-header"><span class="assistant-logo">🤖</span><span class="assistant-name">CognIQ</span><span class="assistant-tag">&middot; grounded answer</span></div><div class="assistant-card"><div class="assistant-body">{msg["content"]}</div>{sources_html}</div><div class="utility-row"><span class="utility-item">📋 Copy</span><span class="utility-item">👍 Helpful</span><span class="utility-item">🔗 Open ticket</span></div></div>', unsafe_allow_html=True)

# --- APPLICATION SCOPE SELECTOR ---
st.selectbox(
    "Application scope",
    APP_OPTIONS,
    key="selected_app",
    format_func=lambda a: f"🎯 {a}",
    label_visibility="collapsed",
    help="Scope every answer to a specific application. This prepends an application "
         "prompt and restricts document retrieval so ambiguous questions "
         "(e.g. 'how is a user created') are answered for the selected app only.",
)

# Resolve the application-scoping prompt and retrieval namespace filter for the current selection.
app_prompt = build_app_prompt(st.session_state["selected_app"])
allowed_namespaces = get_allowed_namespaces(st.session_state["selected_app"])

# --- CHAT INPUT ---
is_offline = bool(st.session_state.get("engine_error") or not st.session_state.get("models_loaded"))
chat_placeholder = "Ask about a ticket, error, or how-to..." if not is_offline else "RAG Engine is currently offline..."

if prompt := st.chat_input(chat_placeholder, disabled=is_offline):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    db.save_message(username, "user", prompt)
    
    st.markdown(f'<div class="user-msg-container"><div class="user-msg-bubble">{prompt}</div></div>', unsafe_allow_html=True)

    # Generate response
    response_placeholder = st.empty()

    with st.status("⚙️ Response Details", expanded=False) as status:
        # 1. Retrieval
        print(f"\n[SYSTEM] Received User Query: {prompt}", flush=True)
        st.write(f"Searching index {'(Hybrid+' if use_hybrid else '('}{'HyDE+' if use_hyde else ''}{'Rerank)' if use_rerank else ')'}...")
        if allowed_namespaces:
            st.write(f"🎯 Scoped to **{st.session_state['selected_app']}** documents only.")
            print(f"[SYSTEM] Retrieval scoped to namespaces: {allowed_namespaces}", flush=True)
        print(f"[SYSTEM] Executing Retrieval Pipeline (Hybrid={use_hybrid}, HyDE={use_hyde}, Rerank={use_rerank})", flush=True)
        top_chunks, metrics = engine.retrieve(
            prompt,
            k=3,
            use_hybrid=use_hybrid,
            use_hyde=use_hyde,
            use_rerank=use_rerank,
            use_parent=use_parent,
            allowed_namespaces=allowed_namespaces
        )
        
        if not top_chunks:
            if allowed_namespaces:
                response = f"I think this info isn't yet added to my knowledge base for {st.session_state['selected_app']}."
            else:
                response = "Not found."
            sources_meta = []
            response_placeholder.markdown(f'<div class="assistant-container"><div class="assistant-header"><span class="assistant-logo">🤖</span><span class="assistant-name">CognIQ</span><span class="assistant-tag">&middot; grounded answer</span></div><div class="assistant-card"><div class="assistant-body">{response}</div></div></div>', unsafe_allow_html=True)
            gen_time = 0
        else:
            st.write(f"Found {len(top_chunks)} relevant segments.")
            context = ""
            sources_meta = []
            for i, c in enumerate(top_chunks):
                meta = c.get("metadata", {})
                meta_header = ""
                if meta:
                    meta_fields = []
                    if meta.get("author"): meta_fields.append(f"Author: {meta['author']}")
                    if meta.get("created_time"): meta_fields.append(f"Created: {meta['created_time']}")
                    if meta.get("modified_time"): meta_fields.append(f"Modified: {meta['modified_time']}")
                    if meta_fields:
                        meta_header = " | " + " | ".join(meta_fields)
                context += f"\n[Source {i+1}: {c['source']}{meta_header}]\n{c['text']}\n"
                
                sources_meta.append({
                    "id": i+1, 
                    "source": c["source"], 
                    "text": c.get("retrieval_text", c["text"]),
                    "score": c.get("score", 0.95 - (i * 0.05)),
                    "metadata": meta
                })
            
            # 2. Generation
            st.write("Synthesizing answer...")
            print(f"[SYSTEM] Generation starting...", flush=True)
            full_response = ""
            
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
                api_key=st.session_state.get("google_api_key"),
                app_prompt=app_prompt
            )

            start_time = time.time()
            for new_text in streamer:
                full_response += new_text
                display_response = strip_fallback_prefix(full_response)
                response_placeholder.markdown(f'<div class="assistant-container"><div class="assistant-header"><span class="assistant-logo">🤖</span><span class="assistant-name">CognIQ</span><span class="assistant-tag">&middot; grounded answer</span></div><div class="assistant-card"><div class="assistant-body">{display_response}▌</div></div></div>', unsafe_allow_html=True)

            gen_time = time.time() - start_time
            print(f"[SYSTEM] Generation finished in {gen_time:.2f}s", flush=True)

            # Strip the fallback phrase if the model emitted it AND then answered anyway
            full_response = strip_fallback_prefix(full_response)

            # Final output with citations
            response_placeholder.markdown(f'<div class="assistant-container"><div class="assistant-header"><span class="assistant-logo">🤖</span><span class="assistant-name">CognIQ</span><span class="assistant-tag">&middot; grounded answer</span></div><div class="assistant-card"><div class="assistant-body">{full_response}</div>{format_sources_html(sources_meta)}</div><div class="utility-row"><span class="utility-item">📋 Copy</span><span class="utility-item">👍 Helpful</span><span class="utility-item">🔗 Open ticket</span></div></div>', unsafe_allow_html=True)
            response = full_response
            
        # Show Telemetry inside the status block
        st.divider()
        col1, col2, col3 = st.columns(3)
        retrieval_time = metrics.get("semantic_time", 0) + metrics.get("keyword_time", 0) + metrics.get("hyde_gen_time", 0)
        col1.metric("Retrieval", f"{retrieval_time:.3f}s")
        col2.metric("Generation", f"{gen_time:.3f}s")
        col3.metric("Total", f"{retrieval_time + gen_time:.3f}s")
        
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

# Centered Tagline footer under input (now rendered inside st.bottom)

import streamlit as st
import time
import os
import database as db

# Initialize database
db.init_db()

FALLBACK_PHRASE = "I think this info isn't yet added to my knowledge base."

def strip_fallback_prefix(text):
    stripped = text.lstrip()
    if stripped.startswith(FALLBACK_PHRASE):
        remainder = stripped[len(FALLBACK_PHRASE):].lstrip(" .\n\r\t")
        if remainder:
            return remainder
    return text

# =====================================================
# 🎯 APPLICATION SCOPING
# =====================================================
DEFAULT_APP = "All Applications"

APP_DESCRIPTIONS = {
    "Trackwise": "TrackWise is a Quality Management System (QMS) used to manage quality processes such as CAPA, deviations, complaints, audits, change control, and quality records.",
    "ThingWorx": "ThingWorx is PTC's Industrial IoT (IIoT) platform used to build and run connected applications, model Things/data, and create dashboards (mashups).",
    "Polarion": "Polarion is Siemens' Application Lifecycle Management (ALM) tool used for requirements management, test management, and work-item tracking.",
    "Windchill GPDM": "Windchill GPDM (Global Product Data Management) is PTC's PLM system used to manage product data, parts, documents, BOMs, and change management.",
}

APP_OPTIONS = [DEFAULT_APP] + list(APP_DESCRIPTIONS.keys())

APP_NAMESPACES = {
    "Trackwise": ["Trackwise"],
    "ThingWorx": ["ThingWorx"],
    "Polarion": ["Polarion"],
    "Windchill GPDM": ["Windchill GPDM & Windchill NA"],
}

def get_allowed_namespaces(app):
    if not app or app == DEFAULT_APP:
        return None
    return APP_NAMESPACES.get(app)

def build_app_prompt(app):
    if not app or app == DEFAULT_APP:
        return None
    description = APP_DESCRIPTIONS.get(app, "")
    return f"APPLICATION CONTEXT: {app} - {description}"

def download_model_ui(repo_id, pattern="q4_k_m.gguf"):
    st.info("[LAYOUT MOCK MODE] Model download is simulated/disabled.")
    return None

# =====================================================
# 🌊 CognIQ Layout Inspector (Mock UI Mode)
# =====================================================

st.set_page_config(page_title="CognIQ Layout Inspector", page_icon="🧠", layout="wide")

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
    padding-top: 5.5rem !important;
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

#MainMenu {display: none !important;}
header {display: none !important;}
footer {display: none !important;}

/* Adjust bottom container layout so app pill sits on left of tagline */
div[data-testid="stBottom"] {
    background-color: #ffffff !important;
    padding-bottom: 12px !important;
}

div[data-testid="stBottom"] > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
}

/* Fixed top header pinning the Application Scope and status on main panel */
.st-key-header_container {
    position: fixed !important;
    top: 0px !important;
    left: 21rem !important;
    right: 0px !important;
    height: 58px !important;
    background-color: #ffffff !important;
    z-index: 99999 !important;
    padding: 8px 2rem !important;
    border-bottom: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
}

section[data-testid="stSidebar"][aria-expanded="false"] ~ section .st-key-header_container {
    left: 0px !important;
}

/* Ensure BaseWeb dropdown menu pops up above fixed elements */
div[data-baseweb="popover"], div[data-baseweb="menu"] {
    z-index: 9999999 !important;
}

/* Tagline centered above the chat input box */
div[data-testid="stChatInput"]::before {
    content: "Grounded in your tickets & KB · every answer cites its source";
    display: block;
    text-align: center;
    font-size: 12px;
    color: #9ca3af;
    margin-bottom: 8px;
    line-height: 28px;
    width: 100%;
}

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
</style>
""", unsafe_allow_html=True)

def format_sources_html(sources):
    if not sources:
        return ""
    chips = []
    for idx, s in enumerate(sources):
        src_name = s.get("source", "") if isinstance(s, dict) else str(s)
        is_ticket = "xlsx" in src_name.lower() or "inc" in src_name.lower()
        badge_class = "badge-ticket" if is_ticket else "badge-kb"
        badge_text = "Ticket" if is_ticket else "KB"
        chip_html = f'<div class="source-chip"><span class="badge {badge_class}">{badge_text}</span><span class="source-text">{src_name}</span><span class="match-pct">95% match</span></div>'
        chips.append(chip_html)
    return f'<hr class="card-divider"/><div class="sources-container"><div class="sources-title">Sources</div>{"".join(chips)}</div>'

# --- INITIALIZE SESSION STATE ---
if "current_model" not in st.session_state:
    st.session_state["current_model"] = "bartowski/Llama-3.2-1B-Instruct-GGUF"
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = True
if "username" not in st.session_state:
    st.session_state["username"] = "harsh"
if "user_email" not in st.session_state:
    st.session_state["user_email"] = "harsh.verma@freseniusmedicalcare.com"
if "user_role" not in st.session_state:
    st.session_state["user_role"] = "admin"
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
if "models_loaded" not in st.session_state:
    st.session_state["models_loaded"] = True
if "engine_error" not in st.session_state:
    st.session_state["engine_error"] = None

username = st.session_state["username"]

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    if os.path.exists("media/combined_logo.png"):
        st.image("media/combined_logo.png", use_container_width=True)
    st.markdown("""
    <div class="sidebar-branding">
        <div class="sidebar-title">CognIQ</div>
        <div class="sidebar-subtitle">advanced RAG engine (Mock Mode)</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("➕ New chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["current_page"] = "chat"
        st.rerun()

    kb_folder = "knowledge_source"
    file_count = 96
    kb_size_str = "51.6 MB"
        
    st.markdown(f"""
    <div class="kb-card">
        <div class="kb-card-title">📂 Knowledge base</div>
        <div class="kb-card-status">
            <span class="status-dot"></span> {file_count:,} documents · {kb_size_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        st.session_state["logged_in"] = False
        st.rerun()

    display_username = st.session_state.get('username', '')
    first_letter = display_username[0].upper() if display_username else 'C'
    st.markdown(f'<div class="profile-container"><div class="profile-avatar">{first_letter}</div><div class="profile-info"><div class="profile-name">{display_username}</div><div class="profile-role">{st.session_state.get("user_role", "").title()}</div></div><div class="profile-chevron">👤</div></div>', unsafe_allow_html=True)

# --- DEDICATED PAGES (MOCKED) ---
if st.session_state.get("current_page") == "manage_kb":
    st.title("📂 Document & Knowledge Base Manager (Mock Mode)")
    if st.button("⬅️ Back to Chat", use_container_width=True, type="secondary"):
        st.session_state["current_page"] = "chat"
        st.rerun()
    st.stop()

if st.session_state.get("current_page") == "manage_users":
    st.title("👥 User Administration Dashboard (Mock Mode)")
    if st.button("⬅️ Back to Chat", use_container_width=True, type="secondary"):
        st.session_state["current_page"] = "chat"
        st.rerun()
    st.stop()

if st.session_state.get("current_page") == "settings":
    st.title("⚙️ RAG Engine Configurations (Mock Mode)")
    if st.button("⬅️ Back to Chat", use_container_width=True, type="secondary"):
        st.session_state["current_page"] = "chat"
        st.rerun()
    st.stop()

# --- SESSION STATE FOR CHAT ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- HEADER & STATUS BANNERS ---
with st.container(key="header_container"):
    col_hdr1, col_hdr2 = st.columns([1, 2.2])
    with col_hdr1:
        st.markdown("""
        <div class="engine-status" style="padding-top: 8px;">
            <span class="status-dot"></span> Engine ready (Mock Mode)
        </div>
        """, unsafe_allow_html=True)
    with col_hdr2:
        st.selectbox(
            "Application Scope",
            APP_OPTIONS,
            key="selected_app",
            format_func=lambda a: f"🎯 Application: {a}",
            label_visibility="collapsed",
            help="Scope every answer to a specific application context."
        )

# Empty-chat landing page logo layout
if not st.session_state["messages"]:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
    with col_logo2:
        if os.path.exists("media/combined_logo.png"):
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

# --- CHAT INPUT ---
chat_placeholder = "Ask about a ticket, error, or how-to..."

if prompt := st.chat_input(chat_placeholder):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-msg-container"><div class="user-msg-bubble">{prompt}</div></div>', unsafe_allow_html=True)

    # Mock Assistant Answer
    mock_answer = f"**[MOCK UI RESPONSE]**\n\nYou asked: *{prompt}*\nSelected application scope: **{st.session_state['selected_app']}**.\n\nThis copy (`ui_mock.py`) has heavy model loading, transformers, embedding calculations, and retrieval pipelines commented/bypassed so you can test and iterate on layout changes instantly."
    mock_sources = [{"source": "Trackwise_CAPA_Guide.pdf"}, {"source": "INC983421_Resolution.xlsx"}]

    response_placeholder = st.empty()
    response_placeholder.markdown(f'<div class="assistant-container"><div class="assistant-header"><span class="assistant-logo">🤖</span><span class="assistant-name">CognIQ</span><span class="assistant-tag">&middot; grounded answer</span></div><div class="assistant-card"><div class="assistant-body">{mock_answer}</div>{format_sources_html(mock_sources)}</div><div class="utility-row"><span class="utility-item">📋 Copy</span><span class="utility-item">👍 Helpful</span><span class="utility-item">🔗 Open ticket</span></div></div>', unsafe_allow_html=True)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": mock_answer,
        "sources": mock_sources
    })



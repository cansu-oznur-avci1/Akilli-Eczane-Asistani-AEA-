import streamlit as st
from main import run_agent, run_agent_stream, generate_chat_title
import os
import pandas as pd
import json
from utils import history_manager as hm
from utils.formatting_utils import format_assistant_response
from utils.i18n import get_text

st.set_page_config(page_title="Akıllı Eczacı Asistanı", page_icon="💊", layout="wide")

if "lang" not in st.session_state:
    st.session_state.lang = "tr"
if "dark_theme" not in st.session_state:
    st.session_state.dark_theme = True

# Dynamic CSS based on theme
theme_css = """
<style>
/* Modern typography */
html, body, [class*="css"]  {
    font-family: 'Inter', 'Roboto', sans-serif;
}

/* Chat message container styling */
.stChatMessage {
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

/* Force consistent text and header sizes in all chat messages */
.stChatMessage h1, .stChatMessage h2, .stChatMessage h3, .stChatMessage h4 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.5rem !important;
    line-height: 1.4 !important;
}

.stChatMessage p, .stChatMessage li, .stChatMessage span {
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

/* Sidebar styling improvements */
.stButton>button {
    border-radius: 8px;
    transition: all 0.2s;
    font-size: 1rem !important;
}
"""

if st.session_state.dark_theme:
    theme_css += """
    [data-testid="stAppViewContainer"] { background-color: #121212 !important; color: #E0E0E0 !important; }
    [data-testid="stSidebar"] { background-color: #1E1E1E !important; color: #E0E0E0 !important; border-right: 1px solid #333; }
    [data-testid="stHeader"] { background-color: #121212 !important; color: #E0E0E0 !important; }
    .stMarkdown, p, h1, h2, h3, h4, span { color: #E0E0E0 !important; }
    
    /* Fix buttons in dark mode */
    .stButton>button { background-color: #2D2D2D !important; color: #E0E0E0 !important; border: 1px solid #444 !important; }
    /* Fix chat input area in dark mode */
    [data-testid="stBottom"], [data-testid="stBottom"] > div { background-color: #121212 !important; }
    [data-testid="stBottomBlockContainer"] { background-color: transparent !important; }
    
    .stChatInputContainer { background-color: transparent !important; border: none !important; }
    
    [data-testid="stChatInput"] { 
        background-color: #2D2D2D !important; 
        border: 1px solid #444444 !important; 
        border-radius: 24px !important; 
        overflow: hidden !important;
    }
    
    [data-testid="stChatInput"] * { 
        background-color: transparent !important; 
        color: #FFFFFF !important; 
        caret-color: #FFFFFF !important;
        border: none !important;
    }
    
    [data-testid="stChatInput"] textarea::placeholder { color: #AAAAAA !important; }
    
    /* Fix status widget in dark mode */
    [data-testid="stStatusWidget"] { background-color: transparent !important; border: none !important; }
    [data-testid="stStatusWidget"] * { background-color: transparent !important; color: #E0E0E0 !important; }
    
    .stChatMessage { background-color: #1E1E1E !important; border: 1px solid #333; }
    """
else:
    theme_css += """
    [data-testid="stAppViewContainer"] { background-color: #FFFFFF !important; color: #1C1C1E !important; }
    [data-testid="stSidebar"] { background-color: #F8F9FA !important; color: #1C1C1E !important; border-right: 1px solid #E5E5E5; }
    [data-testid="stHeader"] { background-color: #FFFFFF !important; color: #1C1C1E !important; }
    .stMarkdown, p, h1, h2, h3, h4, span { color: #1C1C1E !important; }
    
    /* Fix buttons in light mode */
    .stButton>button { background-color: #FFFFFF !important; color: #1C1C1E !important; border: 1px solid #CCC !important; }
    .stButton>button:hover { border-color: #ff4b4b !important; color: #ff4b4b !important; }
    
    /* Fix chat input area in light mode */
    [data-testid="stBottom"], [data-testid="stBottom"] > div { background-color: #FFFFFF !important; }
    [data-testid="stBottomBlockContainer"] { background-color: #FFFFFF !important; }
    
    .stChatInputContainer { background-color: transparent !important; border: none !important; }
    
    [data-testid="stChatInput"] { 
        background-color: #FFFFFF !important; 
        border: 1px solid #CCCCCC !important; 
        border-radius: 24px !important; 
        overflow: hidden !important;
    }
    
    [data-testid="stChatInput"] * { 
        background-color: transparent !important; 
        color: #000000 !important; 
        caret-color: #000000 !important;
        border: none !important;
    }
    
    [data-testid="stChatInput"] textarea::placeholder { color: #666666 !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: #666666 !important; }
    [data-testid="stChatInput"] button { color: #1C1C1E !important; }
    
    /* Fix status widget in light mode */
    [data-testid="stStatusWidget"] { background-color: transparent !important; border: none !important; }
    [data-testid="stStatusWidget"] * { background-color: transparent !important; color: #1C1C1E !important; }
    
    .stChatMessage { background-color: #F8F9FA !important; border: 1px solid #eee; }
    """

theme_css += "</style>"
st.markdown(theme_css, unsafe_allow_html=True)

# Initialize session state for auth
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None

def load_users():
    if not os.path.exists("data/users.json"):
        admin_pass = os.getenv("ADMIN_PASSWORD", "admin123")
        user_pass = os.getenv("USER_PASSWORD", "user123")
        default_users = {
            "admin": {"password": admin_pass, "role": "admin"},
            "user": {"password": user_pass, "role": "user"}
        }
        os.makedirs("data", exist_ok=True)
        with open("data/users.json", "w", encoding="utf-8") as f:
            json.dump(default_users, f)
        return default_users
    else:
        with open("data/users.json", "r", encoding="utf-8") as f:
            return json.load(f)

def save_users(users):
    os.makedirs("data", exist_ok=True)
    with open("data/users.json", "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

def login_page():
    lang = st.session_state.lang
    st.title(get_text(lang, "login_title"))
    
    c1, c2 = st.columns([8, 2])
    with c2:
        lang_btn_text = "🇬🇧 EN" if lang == "tr" else "🇹🇷 TR"
        if st.button(lang_btn_text, use_container_width=True):
            st.session_state.lang = "en" if lang == "tr" else "tr"
            st.rerun()

    tab1, tab2 = st.tabs([get_text(lang, "login_tab"), get_text(lang, "register_tab")])
    
    users = load_users()

    with tab1:
        st.write(get_text(lang, "login_prompt"))
        username = st.text_input(get_text(lang, "username"), key="login_user")
        password = st.text_input(get_text(lang, "password"), type="password", key="login_pass")
        if st.button(get_text(lang, "login_btn")):
            if username in users and users[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.role = users[username]["role"]
                st.session_state.username = username
                st.rerun()
            else:
                st.error(get_text(lang, "invalid_login"))

    with tab2:
        st.write(get_text(lang, "register_prompt"))
        new_username = st.text_input(get_text(lang, "username"), key="reg_user")
        new_password = st.text_input(get_text(lang, "password"), type="password", key="reg_pass")
        if st.button(get_text(lang, "register_btn")):
            if not new_username or not new_password:
                st.error(get_text(lang, "fill_fields"))
            elif new_username in users:
                st.warning(f"'{new_username}' {get_text(lang, 'user_exists')}")
            else:
                users[new_username] = {"password": new_password, "role": "user"}
                save_users(users)
                st.success(get_text(lang, "register_success"))

def user_page():
    lang = st.session_state.lang
    st.title(get_text(lang, "app_title"))
    st.markdown(get_text(lang, "app_subtitle"))

    with st.sidebar:
        # Theme and Lang Controls
        c1, c2 = st.columns(2)
        with c1:
            theme_btn_text = "☀️ Light" if st.session_state.dark_theme else "🌙 Dark"
            if st.button(theme_btn_text, use_container_width=True):
                st.session_state.dark_theme = not st.session_state.dark_theme
                st.rerun()
        with c2:
            lang_btn_text = "🇬🇧 EN" if lang == "tr" else "🇹🇷 TR"
            if st.button(lang_btn_text, use_container_width=True):
                st.session_state.lang = "en" if lang == "tr" else "tr"
                st.rerun()
                
        st.divider()

        st.header(get_text(lang, "chat_history"))
        if st.button(get_text(lang, "new_chat"), use_container_width=True, type="primary"):
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        sessions = hm.get_user_sessions(st.session_state.get("username", "user"))
        for s in sessions:
            col1, col2 = st.columns([5, 1])
            with col1:
                title_disp = s['title']
                if len(title_disp) > 25: title_disp = title_disp[:22] + "..."
                if st.button(f"💬 {title_disp}", key=f"sel_{s['id']}", use_container_width=True):
                    st.session_state.current_session_id = s["id"]
                    st.session_state.messages = hm.get_session_messages(s["id"])
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{s['id']}"):
                    hm.delete_session(s["id"])
                    if st.session_state.get("current_session_id") == s["id"]:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    st.rerun()
                    
        st.divider()
        if st.button(get_text(lang, "logout"), use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.session_state.username = None
            st.session_state.messages = []
            st.session_state.current_session_id = None
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(get_text(lang, "chat_input_placeholder")):
        if not st.session_state.current_session_id:
            # Smart titling
            title = generate_chat_title(prompt, lang)
            st.session_state.current_session_id = hm.create_session(st.session_state.get("username", "user"), title)
            
        hm.save_message(st.session_state.current_session_id, "user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_container = st.status(get_text(lang, "status_thinking"), expanded=True)
            try:
                history = st.session_state.messages[:-1]
                
                final_response = get_text(lang, "error_msg")
                for node_name, state in run_agent_stream(prompt, history, lang):
                    if node_name == "Sorgu_Siniflandirma":
                        status_container.write(get_text(lang, "status_analyze"))
                    elif node_name == "Kural_Motoru_Node":
                        status_container.write(get_text(lang, "status_rules"))
                    elif node_name == "Vektor_Veritabani_RAG":
                        status_container.write(get_text(lang, "status_db"))
                    elif node_name == "Web_Ara_Tavily_Node":
                        status_container.write(get_text(lang, "status_web"))
                    elif node_name == "Asistan_LLM":
                        status_container.write(get_text(lang, "status_generate"))
                        final_response = state.get("yanit", get_text(lang, "error_msg"))
                
                status_container.update(label=get_text(lang, "status_complete"), state="complete", expanded=False)
                formatted_yanit = format_assistant_response(final_response)
                st.markdown(formatted_yanit)
                hm.save_message(st.session_state.current_session_id, "assistant", formatted_yanit)
                st.session_state.messages.append({"role": "assistant", "content": formatted_yanit})
                
            except Exception as e:
                status_container.update(label=get_text(lang, "status_error"), state="error", expanded=False)
                yanit = f"{get_text(lang, 'error_msg')} {str(e)}"
                st.markdown(yanit)
                hm.save_message(st.session_state.current_session_id, "assistant", yanit)
                st.session_state.messages.append({"role": "assistant", "content": yanit})

def admin_page():
    col1, col2 = st.columns([8, 1])
    with col1:
        st.title("⚙️ Admin Paneli")
    with col2:
        if st.button("Çıkış"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.rerun()

    st.header("LLM Ayarları")
    GROQ_MODELS = {
        "llama-3.1-8b-instant": "⚡ Llama 3.1 8B Instant",
        "llama-3.3-70b-versatile": "🦙 Llama 3.3 70B Versatile",
        "openai/gpt-oss-120b": "🤖 GPT OSS 120B",
        "openai/gpt-oss-20b": "🤖 GPT OSS 20B",
    }
    model_display = st.selectbox(
        "LLM Modeli",
        options=list(GROQ_MODELS.keys()),
        format_func=lambda x: GROQ_MODELS[x],
        index=0
    )
    os.environ["GROQ_MODEL"] = model_display

    temp = st.slider("Temperature", 0.0, 1.0, float(os.getenv("GROQ_TEMPERATURE", "0.2")), 0.05)
    os.environ["GROQ_TEMPERATURE"] = str(temp)

    rag_k = st.slider("RAG Chunk Sayısı (k)", 1, 8, 4)
    os.environ["AEA_RAG_K"] = str(rag_k)

    web_search = st.toggle("Web Search Fallback", value=False)
    os.environ["AEA_WEB_SEARCH"] = "1" if web_search else "0"

    st.divider()

    st.header("Kural Tablosu Yönetimi (etkilesimler.csv)")
    csv_path = "data/etkilesimler.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df)

        st.subheader("Yeni Etkileşim Ekle")
        with st.form("add_interaction"):
            c1, c2, c3 = st.columns(3)
            with c1:
                ilac = st.text_input("İlaç Adı")
            with c2:
                madde = st.text_input("Etkileşen Madde")
            with c3:
                risk = st.selectbox("Risk Seviyesi", ["HIGH", "LOW", "NONE", "UNKNOWN"])
            
            if st.form_submit_button("Ekle"):
                if ilac and madde:
                    new_row = pd.DataFrame({"ilac_adi": [ilac], "etkilesen_madde": [madde], "risk_seviyesi": [risk]})
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_csv(csv_path, index=False)
                    st.success("Başarıyla eklendi!")
                    st.rerun()
                else:
                    st.error("Lütfen İlaç Adı ve Etkileşen Maddeyi doldurun.")

        st.subheader("Etkileşim Çıkar")
        with st.form("remove_interaction"):
            remove_idx = st.number_input("Silinecek Satır İndeksi (0'dan başlar)", min_value=0, max_value=len(df)-1 if len(df)>0 else 0, step=1)
            if st.form_submit_button("Sil"):
                if len(df) > 0 and 0 <= remove_idx < len(df):
                    df = df.drop(remove_idx)
                    df.to_csv(csv_path, index=False)
                    st.success("Başarıyla silindi!")
                    st.rerun()
                else:
                    st.error("Geçersiz indeks.")
    else:
        st.warning("CSV dosyası bulunamadı!")

if not st.session_state.logged_in:
    login_page()
elif st.session_state.role == "admin":
    admin_page()
elif st.session_state.role == "user":
    user_page()

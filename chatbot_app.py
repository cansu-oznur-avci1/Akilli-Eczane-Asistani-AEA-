import streamlit as st
from main import run_agent
import os
import pandas as pd
import json

st.set_page_config(page_title="Akıllı Eczacı Asistanı", page_icon="💊")

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
    st.title("Hoş Geldiniz")
    tab1, tab2 = st.tabs(["Giriş Yap", "Kayıt Ol"])
    
    users = load_users()

    with tab1:
        st.write("Lütfen devam etmek için giriş yapın.")
        username = st.text_input("Kullanıcı Adı", key="login_user")
        password = st.text_input("Şifre", type="password", key="login_pass")
        if st.button("Giriş"):
            if username in users and users[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.role = users[username]["role"]
                st.rerun()
            else:
                st.error("Geçersiz kullanıcı adı veya şifre!")

    with tab2:
        st.write("Yeni bir hesap oluşturun.")
        new_username = st.text_input("Kullanıcı Adı", key="reg_user")
        new_password = st.text_input("Şifre", type="password", key="reg_pass")
        if st.button("Kayıt Ol"):
            if not new_username or not new_password:
                st.error("Lütfen kullanıcı adı ve şifre girin.")
            elif new_username in users:
                st.warning(f"'{new_username}' kullanıcısı zaten kayıtlı. Eğer şifrenizi değiştirmek istiyorsanız lütfen yönetici ile iletişime geçin veya 'Giriş Yap' sekmesini kullanın.")
            else:
                users[new_username] = {"password": new_password, "role": "user"}
                save_users(users)
                st.success("Kayıt başarılı! Şimdi 'Giriş Yap' sekmesinden giriş yapabilirsiniz.")

def user_page():
    # Only Chatbot Interface
    st.title("💊 Akıllı Eczacı Asistanı (AEA)")
    st.markdown("İlaç etkileşimleri, yan etkiler ve genel farmakoloji hakkında sorularınızı sorabilirsiniz.")

    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("Çıkış"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Yanıt üretiliyor..."):
                try:
                    history = st.session_state.messages[:-1]
                    response_state = run_agent(prompt, history)
                    yanit = response_state.get("yanit", "Yanıt üretilemedi.")
                    st.markdown(yanit)
                except Exception as e:
                    yanit = f"Bir hata oluştu: {str(e)}"
                    st.markdown(yanit)

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

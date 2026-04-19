import streamlit as st
from main import run_agent
import os

st.set_page_config(page_title="Akıllı Eczacı Asistanı", page_icon="💊")

st.title("💊 Akıllı Eczacı Asistanı (AEA)")
st.markdown("İlaç etkileşimleri, yan etkiler ve genel farmakoloji hakkında sorularınızı sorabilirsiniz.")

# Mesaj geçmişini saklamak için session state kullanımı
if "messages" not in st.session_state:
    st.session_state.messages = []

# Önceki mesajları ekranda gösterme
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni girdi alma
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    # Kullanıcı mesajını ekleme ve gösterme
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistan yanıtını alma ve gösterme
    with st.chat_message("assistant"):
        status_box = st.status("🔍 Sorunuz analiz ediliyor...", expanded=True)
        yanit = "Yanıt üretilemedi."
        response_state = {}
        try:
            with status_box:
                st.write("📋 **1. Adım:** Sorgu kategorize ediliyor...")
                history = st.session_state.messages[:-1]

                # Kategoriyi önceden göster
                from main import (
                    _categorize_query_text,
                    _extract_interaction_entities,
                    _extract_side_effect_entities,
                    _extract_general_entities,
                )
                q_type = _categorize_query_text(prompt)
                type_labels = {
                    "interaction": "💊 İlaç Etkileşimi",
                    "side_effect": "⚠️ Yan Etki",
                    "general_info": "ℹ️ Genel Bilgi",
                    "unknown": "❓ Bilinmiyor",
                }
                st.write(f"   Tür: **{type_labels.get(q_type, q_type)}**")

                st.write("📚 **2. Adım:** Kural motoru & vektör veritabanı aranıyor...")
                response_state = run_agent(prompt, history)
                yanit = response_state.get("yanit", "Yanıt üretilemedi.")

                # Kaynak bilgisi
                rag_ozet = response_state.get("rag_ozet", "")
                if rag_ozet.startswith("[WEB KAYNAĞI"):
                    kaynak = "🌐 Web Araması (DuckDuckGo)"
                elif rag_ozet and len(rag_ozet.strip()) > 10:
                    kaynak = "📗 KÜB/KT Vektör Veritabanı (RAG)"
                else:
                    kaynak = "🔧 Deterministik (Kural Motoru)"

                st.write(f"   Kullanılan Kaynak: **{kaynak}**")
                st.write("🤖 **3. Adım:** Yanıt üretiliyor (LLM)...")

            status_box.update(label="✅ Yanıt hazır!", state="complete", expanded=False)

            # Geliştirici özeti
            with st.expander("🛠️ Geliştirici Özeti (Sistem Çıktıları)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Sorgu Türü:** {response_state.get('query_type', '—')}")
                    st.write(f"**Tespit Edilen İlaç:** {response_state.get('ilac_adi', '—') or '—'}")
                    st.write(f"**İlgili Madde/Semptom:** {response_state.get('etkilesen_madde', '') or response_state.get('yan_etki', '') or '—'}")
                with col2:
                    st.write(f"**Risk Seviyesi:** {response_state.get('risk_seviyesi', '—')}")
                    st.write(f"**Kaynak:** {kaynak}")
                    rag_chunks = response_state.get("rag_chunks", [])
                    st.write(f"**RAG Chunk Sayısı:** {len(rag_chunks)}")
                # RAG chunk kaynakları — hangi PDF'den geldi?
                if rag_chunks:
                    st.markdown("**RAG Chunk Kaynakları:**")
                    for i, ch in enumerate(rag_chunks[:4]):
                        import os as _os
                        src = response_state.get('rag_ozet', '')
                        # chunk text önizlemesi
                        preview = ch[:120].replace("\n", " ") if ch else "—"
                        st.caption(f"`Chunk {i+1}:` {preview}...")

            st.markdown(yanit)

        except Exception as e:
            status_box.update(label="❌ Hata oluştu", state="error", expanded=False)
            yanit = f"Bir hata oluştu: {str(e)}"
            st.markdown(yanit)

    # Asistan yanıtını mesaj geçmişine ekleme
    st.session_state.messages.append({"role": "assistant", "content": yanit})


# Sidebar — Admin Paneli
with st.sidebar:
    st.header("⚙️ Admin Paneli")
    
    admin_pass = st.text_input("Admin Şifresi", type="password")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
    
    if admin_pass == ADMIN_PASSWORD:
        st.success("Admin girişi başarılı")
        
        # Model seçimi
        GROQ_MODELS = {
            # ── Production ──────────────────────────────────────────────
            "llama-3.1-8b-instant":             "⚡ Llama 3.1 8B Instant [Production]",
            "llama-3.3-70b-versatile":          "🦙 Llama 3.3 70B Versatile [Production]",
            "openai/gpt-oss-120b":              "🤖 GPT OSS 120B [Production]",
            "openai/gpt-oss-20b":               "🤖 GPT OSS 20B [Production]",
            # ── Preview (deneysel) ──────────────────────────────────────
            "meta-llama/llama-4-scout-17b-16e-instruct": "🔬 Llama 4 Scout 17B [Preview]",
            "qwen/qwen3-32b":                   "🔬 Qwen3 32B [Preview]",
        }
        model_display = st.selectbox(
            "LLM Modeli",
            options=list(GROQ_MODELS.keys()),
            format_func=lambda x: GROQ_MODELS[x],
            index=0
        )
        os.environ["GROQ_MODEL"] = model_display
        
        # Temperature
        temp = st.slider("Temperature", 0.0, 1.0, 
                         float(os.getenv("GROQ_TEMPERATURE", "0.2")), 0.05)
        os.environ["GROQ_TEMPERATURE"] = str(temp)
        
        # RAG k değeri
        rag_k = st.slider("RAG Chunk Sayısı (k)", 1, 8, 4)
        os.environ["AEA_RAG_K"] = str(rag_k)
        
        # Web search toggle
        web_search = st.toggle("Web Search Fallback", value=False)
        os.environ["AEA_WEB_SEARCH"] = "1" if web_search else "0"
        
        st.divider()
        st.caption(f"Model: {model_display} | Temp: {temp} | k: {rag_k}")
    elif admin_pass:
        st.error("Yanlış şifre")

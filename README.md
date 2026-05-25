# 💊 Akıllı Eczacı Asistanı (AEA)

Akıllı Eczacı Asistanı (AEA), eczacılara ilaç etkileşimleri, kullanım talimatları ve yan etkiler konusunda destek sağlayan, hibrit bir **Kural Motoru + RAG (Retrieval-Augmented Generation)** sistemidir.

Bu uygulama, hem deterministik (kural tabanlı) kontrolleri hem de LLM destekli kanıta dayalı (KÜB/KT belgeleri üzerinden) yanıtları birleştirerek güvenli ve açıklanabilir bir asistan deneyimi sunar.

---

## 🚀 Öne Çıkan Özellikler

-   **Deterministik Kural Motoru:** Risk puanı ve kritik etkileşimler için önceden tanımlanmış veri tabanını kullanarak %100 doğruluk sağlar.
-   **Gelişmiş RAG Sistemi:** KÜB (Kısa Ürün Bilgisi) ve KT (Kullanma Talimatı) PDF'leri üzerinden bağlamsal bilgi çıkarımı.
-   **Self-Correction (Halisünasyon Kontrolü):** Reflexion Node ile model çıktıları yayınlanmadan önce risk çelişkileri ve uydurma bilgilere (halisünasyon) karşı otonom olarak denetlenir ve gerekirse model kendi cevabını düzeltir.
-   **Modern ve Profesyonel Arayüz:** Özel CSS mimarisi ile kusursuz çalışan **Karanlık (Dark) ve Aydınlık (Light) Tema** desteği. Giriş ekranı ve yerleşik ayarlar paneli (Settings modal) de dahil olmak üzere her arayüz aşaması, yüksek kontrastlı ve erişilebilir özel CSS seçicileriyle (örneğin `:first-of-type` checkbox entegrasyonu, modal başlık görünürlük kuralları) optimize edilmiştir.
-   **Akıllı Sohbet Geçmişi:** Yapay zeka tarafından otomatik isimlendirilen, SQLite tabanlı kalıcı sohbet geçmişi.
-   **Çoklu Dil Desteği (TR/EN):** Tek tuşla kullanıcı arayüzünü ve yapay zeka çıktılarını İngilizce veya Türkçe'ye çevirme.
-   **Çok Sayfalı ve Güvenli Yapı (RBAC):** Admin ve User rolleri ile yetkilendirilmiş erişim kontrolü, şifre değiştirme ve kayıt yönetim sistemi.
-   **Gelişmiş Admin Paneli:** Model parametrelerini değiştirme, kural tablosunu düzenleme ve arayüz üzerinden kod yazmadan KÜB/KT belgelerini (PDF) yükleyip vektör veritabanını oluşturma.
-   **Web Arama Desteği:** Yerel belgelerde bulunmayan bilgiler için arama motoru entegrasyonu.

---

## 🏗 Mimari Yapı

Sistem üç ana katmandan oluşur:
1.  **Rule Engine (`engine/`):** Kritik etkileşimler için CSV tabanlı hızlı kontrol.
2.  **Vector DB (`vector_db/`):** PDF belgelerinin vektörleştirilmesi ve anlamsal arama.
3.  **Agent Logic (`backend/`):** LangGraph tabanlı iş akışı yönetimi ve karar mekanizması.

Sistemin çalışma akış grafiği:
![LangGraph İş Akışı](AEA_flow.png)

---

## 📦 Kurulum

1.  **Depoyu klonlayın ve bağımlılıkları yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Çevresel değişkenleri yapılandırın:**
    `.env.example` dosyasını `.env` olarak kopyalayın ve API anahtarlarınızı girin.
    ```bash
    cp .env.example .env
    ```

3.  **PDF Belgelerini İşleyin (RAG için):**
    `pdfs/` klasörüne KÜB/KT dosyalarını yerleştirin ve vektör veritabanını oluşturun:
    ```bash
    python -m vector_db.ingest_data --input-dir pdfs --clear
    ```

---

## 💻 Kullanım

### 🎨 Chatbot Arayüzü (Önerilen)
Modern Streamlit arayüzünü başlatmak için:
```bash
streamlit run chatbot_app.py
```

### ⌨️ CLI (Komut Satırı)
Doğrudan terminal üzerinden soru sormak için:
```bash
python main.py "Enapril ile Alkol etkileşimi nedir?"
```

---

## 📊 Veri Yönetimi

-   **Kural Tablosu (`data/etkilesimler.csv`):** İlaçlar arası etkileşimleri ve risk seviyelerini tanımlar.
-   **Vektör Deposu:** `vector_db/chroma/` dizininde tutulur ve belgelerin semantik indekslerini barındırır.

---

## 🔍 Gözlemlenebilirlik

Sistem, LangChain ekosistemi ile tam uyumludur. `.env` dosyasında LangSmith API anahtarınızı aktif ederek:
-   Girdi/Çıktı kalitesini ölçebilir,
-   Hata ayıklama süreçlerini hızlandırabilir,
-   RAG performansını (Hit Rate, Faithfulness) takip edebilirsiniz.

---

## ⚠️ Yasal Uyarı

Bu sistem bir **karar destek arayüzüdür**. Üretilen yanıtlar tıbbi tavsiye niteliği taşımaz. Nihai karar her zaman uzman bir sağlık profesyoneli (Doktor veya Eczacı) tarafından verilmelidir.


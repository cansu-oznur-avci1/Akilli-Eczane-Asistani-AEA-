# Akıllı Eczacı Asistanı (AEA) — İskelet

Bu repo, **kural motoru (deterministik)** + **RAG (açıklayıcı)** + **LLM (sadeleştirici)** mimarisini temel alan bir başlangıç iskeletidir.

## Klasörler

- `data/`: Kural tablosu (CSV/JSON) gibi kesin veri kaynakları
- `engine/`: Kural motoru ve çekirdek mantık
- `backend/`: LangGraph akışı, LLM bağlantısı ve uygulama girişleri
- `notebooks/`: Deney/analiz defterleri (opsiyonel)

## Hızlı başlatma

1) Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

2) `.env` oluşturun:

```bash
cp .env.example .env
```

3) Çalıştırın:

```bash
python -m backend.app
```

> Not: LLM çağrısı için `GROQ_API_KEY` gerekir. Kural motoru tek başına çalışır.


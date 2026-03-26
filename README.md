# Akıllı Eczacı Asistanı (AEA) — İskelet

Bu repo, **kural motoru (deterministik)** + **RAG (açıklayıcı)** + **LLM (sadeleştirici)** mimarisini temel alan bir başlangıç iskeletidir.

## Klasörler

- `data/`: Kural tablosu (CSV/JSON) gibi kesin veri kaynakları
- `engine/`: Kural motoru ve çekirdek mantık
- `backend/`: LangGraph akışı, LLM bağlantısı ve uygulama girişleri
- `notebooks/`: Deney/analiz defterleri (opsiyonel)

## Hızlı Başlatma (AEA Ajanı)

1) Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

2) `.env` oluşturun:

```bash
cp .env.example .env
```

3) Çalıştırın (bir soru vererek):

```bash
python -m backend.app "Warfarin ile greyfurt suyu etkileşimi nedir?"
```

> Not: LLM çağrısı için `GROQ_API_KEY` gerekir. Vector DB henüz ingest edilmediyse RAG kanıtı boş döner, ama akış yine çalışır.

## PDF ingest (RAG için)

1) KÜB/KT PDF’lerini `pdfs/` klasörüne koyun.
   - Dosya adında genellikle `KÜB` / `KT` bilgisi yer alırsa `doc_type` metadata’sı otomatik çıkarılır.
   - (Örnek dosya adları: `...-kt.pdf`, `...-kub.pdf`)
2) ChromaDB’ye kaydetmek için:

```bash
python -m vector_db.ingest_data --input-dir pdfs --clear
```

> Not: Kalıcı vektör veritabanı `vector_db/chroma/` altında oluşur; repo’ya commit edilmez.

## PDF yükleme noktası (sonradan ekleme)

PDF’leri daha sonra eklediğinizde sadece bu klasöre yeni dosyaları koyun, ardından ingest komutunu tekrar çalıştırın (gerekirse `--clear` olmadan).

## Kural tablosu (`data/etkilesimler.csv`)

Bu dosya şu an demo amaçlı örnektir (etkileşimler için). Siz ileride, toplanan KÜB/KT metinlerinden çıkarılan gerçek kurallarla burayı güncelleyeceksiniz.


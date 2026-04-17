from __future__ import annotations

import re
import os
from typing import List, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages import AIMessage

from engine.rule_engine import QueryType, RiskLevel, RuleEngine
from vector_db.retrieval import format_evidence, retrieve
from backend.llm.groq_client import build_chat_model

# ── Gözlemlenebilirlik (Observability) ──────────────────────────────────────────────
# Lazy singleton: handler'lar modül import'unda DEĞİL, ilk kullanımda başlatılır.
# Bu, import sırasında ağ bağlantısı açılmasını ve uygulamanın donmasını engeller.
_langfuse_handler = None
_langsmith_handler = None
_observability_initialized = False


def _get_observability_callbacks() -> list:
    """Observability handler'larını lazy olarak başlat ve döndür."""
    global _langfuse_handler, _langsmith_handler, _observability_initialized
    if _observability_initialized:
        return [h for h in [_langfuse_handler, _langsmith_handler] if h is not None]

    try:
        from langfuse.langchain import CallbackHandler as LangfuseHandler
        _langfuse_handler = LangfuseHandler()
    except Exception:
        _langfuse_handler = None

    try:
        # langchain_core >= 0.1.x — doğru import yolu
        from langchain_core.tracers.langchain import LangChainTracer
        _langsmith_handler = LangChainTracer()
    except Exception:
        _langsmith_handler = None

    _observability_initialized = True
    return [h for h in [_langfuse_handler, _langsmith_handler] if h is not None]


class AgentState(TypedDict, total=False):
    question: str
    messages: list
    query_type: Literal["interaction", "side_effect", "general_info", "unknown"]

    ilac_adi: str
    etkilesen_madde: str
    yan_etki: str

    risk_seviyesi: str

    rag_chunks: List[str]
    rag_ozet: str
    yanit: str


_INTERACTION_WORDS = [
    "etkileşim",
    "etkilesim",
    "etkileşimi",
    "etkilesimi",
    "birlikte",
    "aynı anda",
    "eş zamanlı",
    "birlikte al",
    "ilaç-ilaç",
    "ilaç-besin",
    "tüketebilir", "tüketilir", "alınabilir", "kullanılabilir mi",
    "beraber", "kombine",
]

_SIDE_EFFECT_WORDS = [
    "yan etki",
    "yanetki",
    "yan etkisi",
    "zarar",
    "advers",
    "ne olur",
]

_GENERAL_WORDS = [
    "nedir",
    "kullanım",
    "kullanımı",
    "kullanılır",
    "ne için",
    "niçin",
    "doz",
    "endikasyon",
    "kontrendikasyon",
    "uyarı",
    "etken madde",
    "farmakoloji",
    "özellik",
    "genel bilgi",
    "ne için",
    "ne işe",
    "nasıl kullan",
    "dozaj",
    "muadil",
    "eşdeğer",
    "prospektüs",
    "ne zaman",
    "kimler kullan",
    "içeriği",
    "içerir",
    "kullanılır",
    "kullanılabilir",
    "hakkında bilgi",
    "bilgi ver",
]


def _normalize_entity(s: str) -> str:
    s = " ".join((s or "").strip().split())
    s = s.strip(".,;:!?()[]{}\"'“”‘’")
    return s


def _categorize_query_text(q: str) -> Literal["interaction", "side_effect", "general_info", "unknown"]:
    q = q.casefold()
    if any(w in q for w in _SIDE_EFFECT_WORDS):
        return "side_effect"
    if any(w in q for w in _INTERACTION_WORDS):
        return "interaction"
    if any(w in q for w in _GENERAL_WORDS):
        return "general_info"
    # fallback: treat as general info
    return "general_info"


def _extract_interaction_entities(q: str) -> tuple[str, str]:
    q_norm = q.replace("\n", " ").strip()

    # "X Y ile kullanılabilir/tüketilebilir/alınabilir mi" formatı
    # Örn: "levotiroksin kalsiyum takviyeleri ile kullanılabilir mi"
    m_sonda_ile = re.search(
        r"^(?P<a>\S+)\s+(?P<b>.+?)\s+ile\s+(?:kullanıl|tüketil|alın|birlikte)",
        q_norm, flags=re.IGNORECASE
    )
    if m_sonda_ile:
        a = _normalize_entity(m_sonda_ile.group("a"))
        b = _normalize_entity(m_sonda_ile.group("b"))
        if a and b and len(a) >= 2 and len(b) >= 2:
            return a, b

    # Common patterns:
    # "X ile Y etkileşimi" / "X ve Y etkileşimi" / "X ile Y birlikte"
    patterns = [
        r"(?P<a>.+?)\s+(?:ile|ve)\s+(?P<b>.+?)\s+etkile\w*",
        r"(?P<a>.+?)\s+(?:ile|ve)\s+(?P<b>.+?)\s+(?:birlikte|aynı anda|eş zamanlı)",
        r"(?P<a>.+?)\s+(?:ile|birlikte)\s+(?P<b>.+)$",
        # YENİ — "X Y ile birlikte kullanılabilir mi" → a=X, b=Y (çok kelimeli ilaç adları için)
        r"(?P<a>[A-Za-zÇçĞğİıÖöŞşÜüa-z]+(?:\s+[A-Za-zÇçĞğİıÖöŞşÜüa-z]+)*?)\s+(?P<b>.+?)\s+ile\s+birlikte",
        # YENİ — "X ile Y kullanılabilir mi"
        r"(?P<a>.+?)\s+ile\s+(?P<b>.+?)\s+kullanıl",
]
    for pat in patterns:
        m = re.search(pat, q_norm, flags=re.IGNORECASE)
        if not m:
            continue
        a = _normalize_entity(m.group("a"))
        b = _normalize_entity(m.group("b"))
        # minimal sanity
        if a and b and len(a) <= 80 and len(b) <= 80:
            return a, b

    if " ile " in q_norm.casefold():
        m_once = re.search(
            r"^(?P<a>.+?)\s+ile\s+(?P<b>.+?)\s+birlikte",
            q_norm, flags=re.IGNORECASE
        )
        if m_once:
            a = _normalize_entity(m_once.group("a"))
            b = _normalize_entity(m_once.group("b"))
            if a and b:
                return a, b

        m_std = re.search(
            r"^(?P<a>.+?)\s+ile\s+(?P<b>.+?)(?:\s+birlikte|\s+kullanıl|\s+tüket|\s+alın|\s*$)",
            q_norm, flags=re.IGNORECASE
        )
        if m_std:
            a = _normalize_entity(m_std.group("a"))
            b = _normalize_entity(m_std.group("b"))
            if a and b:
                return a, b

    return "", ""


def _extract_side_effect_entities(q: str) -> tuple[str, str]:
    q_norm = q.replace("\n", " ").strip()
    # "X’in yan etkisi", "X'in yan etkileri", "X yan etkileri"
    m = re.search(
        r"(?P<ilac>.+?)\s*(?:’|')?(?:in|ın|un|ün|nin|nın|nun|nün)?\s+yan\s+etk\w*",
        q_norm,
        flags=re.IGNORECASE,
    )
    if m:
        return _normalize_entity(m.group("ilac")), ""
    # "X ile birlikte yan etki yapar mı" not enough structure; just extract ilac via first "ile"
    if " ile " in q_norm.casefold():
        parts = re.split(r"\bile\b", q_norm, flags=re.IGNORECASE)
        return _normalize_entity(parts[0]), ""
    return "", ""


def _extract_general_entities(q: str) -> str:
    q_norm = q.replace("\n", " ").strip()
    # "X ne için kullanılır", "X nedir" gibi sorulardan X'i algılama
    m = re.search(
        r"(?P<ilac>.+?)\s+(?:ne için|nedir|nasıl|niçin|kullanım|kullanılır|doz|endikasyon|kontrendikasyon|yan etki)",
        q_norm,
        flags=re.IGNORECASE,
    )
    if m:
        return _normalize_entity(m.group("ilac"))
    
    # Çok kısa sorgularda (1-3 kelime) muhtemelen doğrudan ilaç adı veya arama terimi girilmiştir.
    words = q_norm.split()
    if len(words) <= 3 and not any(w in q_norm.lower() for w in ["nedir", "için", "ne"]):
        return _normalize_entity(q_norm)
        
    return ""


def categorize_query(state: AgentState) -> AgentState:
    q = state.get("question", "") or ""
    messages = state.get("messages", [])
    query_type = state.get("query_type") or _categorize_query_text(q)

    out: AgentState = dict(state)
    out["query_type"] = query_type

    if query_type == "interaction" and not (out.get("ilac_adi") and out.get("etkilesen_madde")):
        ilac, madde = _extract_interaction_entities(q)
        out["ilac_adi"] = ilac
        out["etkilesen_madde"] = madde

    if query_type == "side_effect" and not out.get("ilac_adi"):
        ilac, yan_etki = _extract_side_effect_entities(q)
        out["ilac_adi"] = ilac
        out["yan_etki"] = yan_etki

    if query_type == "general_info" and not out.get("ilac_adi"):
        ilac = _extract_general_entities(q)
        out["ilac_adi"] = ilac

    # Geçmiş bağlam (chat history) ile zenginleştirme
    def needs_history_fallback(q_text, ext_ilac):
        if ext_ilac:
            cues = {"peki", "başka", "bunun", "bununla", "ilaç", "ilaçla", "onunla", "diğer", "bu", "şu", "o", "vitamin", "onu"}
            words = set(ext_ilac.lower().split())
            if words.intersection(cues):
                return True
            return False
            
        if not ext_ilac:
            cues = {"yan etki", "peki", "başka", "birlikte", "bunun", "etkileşim", "onu", "doz", "kullanım", "kullanılır", "nedir", "için", "nasıl"}
            # String içinde geçmesi yetenler (yan etki gibi öbekler için split yeterli olmayabilir, o yüzden "yan etki" in q_text kontrolü de eklendi)
            if any(c in q_text.lower() for c in cues) or "ile " in q_text.lower():
                symptom_cues = ["ağrı", "akıntı", "ateş", "öksürük", "mide", "bulantı", "hastalık", "yanıyor", "var", "oldum", "kötü"]
                if any(s in q_text.lower() for s in symptom_cues):
                    return False
                return True
            if len(q_text.split()) <= 3:
                return True
        return False

    if needs_history_fallback(q, out.get("ilac_adi")):
        prev_ilac = ""
        prev_madde = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                prev_q = m.get("content", "")
                ptype = _categorize_query_text(prev_q)
                pm = ""
                if ptype == "interaction":
                    pi, pm = _extract_interaction_entities(prev_q)
                elif ptype == "side_effect":
                    pi, _ = _extract_side_effect_entities(prev_q)
                else:
                    pi = _extract_general_entities(prev_q)
                
                if pi and not pi.lower().startswith("peki") and not any(w in pi.lower() for w in ["peki", "başka", "bunun", "ilaç", "bu", "şu", "o"]):
                    prev_ilac = pi
                    prev_madde = pm
                    break
        
        if prev_ilac:
            out["ilac_adi"] = prev_ilac
        if prev_madde:
            # Sadece kullanıcı genel/diğer bir etkileşim aramıyorsa 2. maddeyi geçmişten taşı
            if not any(w in q.lower() for w in ["diğer", "başka", "tüm", "genel", "farklı"]):
                out["etkilesen_madde"] = prev_madde

    # general_info/unknown: extraction is intentionally best-effort; retrieval uses the whole question
    return out


def check_rules(state: AgentState) -> AgentState:
    out: AgentState = dict(state)
    query_type = out.get("query_type", "unknown")

    try:
        engine = RuleEngine()
        if query_type == "interaction":
            ilac = out.get("ilac_adi", "")
            madde = out.get("etkilesen_madde", "")
            if ilac and madde:
                rec = engine.lookup_typed(ilac, madde, QueryType.INTERACTION)
                risk = rec.risk_seviyesi if rec else RiskLevel.UNKNOWN
                out["risk_seviyesi"] = risk.value
            else:
                out["risk_seviyesi"] = RiskLevel.UNKNOWN.value
        else:
            out["risk_seviyesi"] = RiskLevel.UNKNOWN.value
    except (FileNotFoundError, ValueError) as e:
        print(f"[RuleEngine] Uyarı: {e}")
        out["risk_seviyesi"] = RiskLevel.UNKNOWN.value

    return out


def get_explanation(state: AgentState) -> AgentState:
    out: AgentState = dict(state)
    question = out.get("question", "") or ""
    query_type = out.get("query_type", "unknown")

    # ── KİMLİK KONTROLÜ ──────────────────────────────────────────
    KIMLIK_KELIMELERI = [
        "kimsin", "kim yaptı", "kim icat", "kim geliştirdi", "kim kurdu", "seni kim",
        "ne yaparsın", "kendin tanıt", "hakkında bilgi ver",
        "who are you", "who made you", "who invented you", "who created you", "who built you",
        "about you", "what do you do", "introduce yourself"
    ]
    if any(k in question.casefold() for k in KIMLIK_KELIMELERI):
        out["rag_chunks"] = []
        out["rag_ozet"] = ""
        out["yanit"] = (
            "Ben Prof. Dr. Ramazan Katırcı danışmanlığında "
            "Bilgisayar Mühendisliği öğrencileri (Cansu Öznur Avcı, Asya Mina Atik, Elifnur Şimşek) ekibi tarafından "
            "geliştirilen Akıllı Eczacı Asistanıyım (AEA). "
            "İlaç etkileşimleri, yan etkiler ve genel farmakoloji konularında "
            "yardımcı olabilirim."
        )
        return out

    risk_seviyesi = out.get("risk_seviyesi", RiskLevel.UNKNOWN.value)

    # Decide whether RAG is needed.
    # - interaction: fetch evidence for HIGH/LOW/UNKNOWN, skip for NONE
    # - side_effect/general_info: always fetch
    do_rag = True
    if query_type == "interaction" and risk_seviyesi == RiskLevel.NONE.value:
        do_rag = False

    # Deterministic response for NONE (no LLM, no RAG evidence needed).
    if query_type == "interaction" and not do_rag:
        out["yanit"] = (
            f"Kural motoru risk seviyesi: {risk_seviyesi}\n"
        )
        out["rag_chunks"] = []
        out["rag_ozet"] = ""
        return out

    evidence_text = ""
    evidence_chunks: List[str] = []

    if do_rag:
        if query_type == "interaction":
            retrieval_query = f"{out.get('ilac_adi','')} {out.get('etkilesen_madde','')}".strip()
            if not retrieval_query:
                retrieval_query = question
        elif query_type == "side_effect":
            ilac = out.get('ilac_adi', '').strip()
            yan = out.get('yan_etki', '').strip()
            if not yan:
                yan = "yan etki"
            retrieval_query = f"{ilac} {yan}".strip()
        else:
            ilac = out.get("ilac_adi", "").strip()
            if ilac and ilac.lower() not in question.lower():
                retrieval_query = f"{ilac} {question}"
            else:
                retrieval_query = question

        k_value = int(os.getenv("AEA_RAG_K", "4"))
        # NOT: doc_type filtresi kaldırıldı — bazı ilaçlar (Aferin Forte gibi) KUB
        # etiketi taşımayabilir, filtrelemek bilgi kaçırılmasına neden oluyordu.
        chunks = retrieve(retrieval_query, k=k_value)
        evidence_chunks = [c.text for c in chunks]
        evidence_text = format_evidence(chunks, max_chars=4200)

        # ── İlaç adı alaka kontrolü ──────────────────────────────────────────
        # ChromaDB semantik benzerlik ile başka ilaçların chunk'larını dönebilir.
        # Eğer ilaç adı chunk'ların hiçbirinde AÇIKÇA geçmiyorsa, bu chunk'ları
        # "kanıt yok" olarak say ve web search fallback'e geç.
        ilac_check = (out.get("ilac_adi") or "").strip().casefold()
        if ilac_check and evidence_text and len(ilac_check) >= 3:
            combined_text = " ".join(evidence_chunks).casefold()
            # İlaç adının herhangi bir token'ı geçiyor mu kontrol et
            ilac_tokens = [t for t in ilac_check.split() if len(t) >= 3]
            found_any = any(tok in combined_text for tok in ilac_tokens) if ilac_tokens else True
            if not found_any:
                if os.getenv("AEA_DEBUG_RAG", "").strip().lower() in ("1", "true"):
                    print(f"[RAG] İlaç '{ilac_check}' chunk'larda bulunamadı, web fallback'e geçiliyor.")
                evidence_text = ""
                evidence_chunks = []
    else:
        evidence_text = ""

    out["rag_chunks"] = evidence_chunks
    out["rag_ozet"] = evidence_text

    # Safety: RAG kanıtı yoksa LLM'e hiç çağrı yapmıyoruz.
    # Böylece uydurma içerik üretme riskini düşürürüz.
    if do_rag and not (evidence_text or "").strip():
        risk = out.get("risk_seviyesi", RiskLevel.UNKNOWN.value)

        # AEA_WEB_SEARCH toggle'ı kontrol et (admin panelinden açılıp kapatılabilir)
        web_search_enabled = os.getenv("AEA_WEB_SEARCH", "0") == "1"
        web_evidence = ""
        if web_search_enabled:
            from vector_db.web_search import web_search_fallback
            web_evidence = web_search_fallback(question)

        if web_evidence:
            evidence_text = f"[WEB KAYNAĞI - KÜB/KT değil]\n{web_evidence}"
            out["rag_chunks"] = []
            out["rag_ozet"] = evidence_text
        else:

            if query_type == "interaction":
                if risk == RiskLevel.HIGH.value:
                    out["yanit"] = (
                        "RİSK SEVİYESİ: HIGH\n\n"
                        "Bu ilaç kombinasyonu yüksek riskli olarak değerlendirilmektedir. "
                        "Elimizdeki KÜB/KT kaynaklarında bu kombinasyona ait detaylı bilgi bulunamamıştır."
                    )
                elif risk == RiskLevel.LOW.value:
                    out["yanit"] = (
                        "RİSK SEVİYESİ: LOW\n\n"
                        "Bu kombinasyon düşük riskli olarak değerlendirilmektedir. "
                        "KÜB/KT kaynaklarında ek detay bulunamamıştır."
                    )
                elif risk == RiskLevel.NONE.value:
                    out["yanit"] = (
                        "Bu kombinasyon için bilinen bir etkileşim risk kaydı bulunmamaktadır."
                    )
                else:  # UNKNOWN
                    out["yanit"] = (
                        "Bu kombinasyon hakkında elimizdeki kaynaklarda bilgi bulunamadı."
                    )
            else:
                out["yanit"] = (
                    "Bu konu hakkında elimizdeki KÜB/KT kaynaklarında bilgi bulunamadı."
                )
            return out
    # LLM part (Groq) - must not change risk decision; only summarize evidence in plain Turkish.

    try:
        model = build_chat_model()
        if query_type == "interaction":
            yanit_formati = (
                "YANIT FORMATI:\n"
                "- Önce kural motoru risk seviyesini ve kısa gerekçesini belirt.\n"
                "- Sonra KÜB/KT kanıtlarına dayalı açıklamayı yap.\n"
                "- HIGH risk varsa ne yapılması gerektiğini belirt.\n"
            )
            risk_input = f"KURAL MOTORU RİSK SEVİYESİ: {risk_seviyesi}\n\n"
        else:
            yanit_formati = (
                "YANIT FORMATI:\n"
                "- GİRİŞ CÜMLESİ VEYA SELAMLAMA YAPMA. Doğrudan sağlanan kanıtlara dayalı net açıklamayı yazarak başla.\n"
                "- Başlıkta veya metinde hiçbir şekilde 'Risk Seviyesi' yazma.\n"
            )
            risk_input = ""

        system_prompt = (
            "GÖREVİN: Sen Prof. Dr. Ramazan Katırcı danışmanlığında, Bilgisayar Mühendisliği öğrencileri "
            "(Cansu Öznur Avcı, Asya Mina Atik, Elifnur Şimşek) tarafından geliştirilen "
            "Akıllı Eczacı Asistanısın (AEA).\n\n"

            "KATÎ KURALLAR:\n\n"

            "1. RİSK KARARI DEĞİŞTİRME YASAKTIR: "
            "Kural motoru 'HIGH' dediyse 'etkileşim bulunmamaktadır' veya 'direkt etkileşim yoktur' YAZAMAZSIN. "
            "Kural motoru 'UNKNOWN' dediyse 'güvenlidir', 'kullanabilirsiniz' veya benzeri onaylayıcı ifadeler YAZAMAZSIN. "
            "Eğer sağlanan kanıtlarda (KÜB/KT veya Web) ilgili konuya veya etkileşime dair hiçbir detay veya uyarı yoksa 'Bu konuda elimdeki kaynaklarda bilgi bulunamadı.' de. "
            "Fakat Kural Motoru UNKNOWN dese bile, eğer sağlanan kanıtlarda sorulan hastalık (örn: atrial fibrilasyon), semptom veya "
            "ilaç kombinasyonlarıyla ilgili kontrendikasyon, uyarı veya etkileşim bilgisi AÇIKÇA varsa, YALNIZCA KANITTAKİ bu bilgileri listele.\n\n"

            "2. SELAMLAMA YASAKTIR: "
            "Yanıtın başında veya sonunda 'Merhaba', 'Ben AEA' gibi ifadeler KULLANMA. "
            "Doğrudan konuya gir. "
            "YALNIZCA kullanıcı 'kimsin' veya 'seni kim yaptı' diye sorarsa: "
            "'Ben Prof. Dr. Ramazan Katırcı danışmanlığında (Cansu Öznur Avcı, Asya Mina Atik, Elifnur Şimşek) "
            "ekibi tarafından geliştirilen Akıllı Eczacı Asistanıyım (AEA).' diyerek tanıt.\n\n"

            "3. DİL EŞLEŞTİRME: "
            "Kullanıcı hangi dilde yazıyorsa AYNI dilde yanıtla. "
            "Türkçe soru → Türkçe yanıt. İngilizce soru → İngilizce yanıt. "
            "Dili asla değiştirme.\n\n"

            "4. ALAN KONTROLÜ VE SEMPTOM SORULARI: "
            "Kullanıcı bir semptom veya hastalık belirtip 'ne kullanabilirim', 'ne iyi gelir' diyorsa, "
            "asla 'şu ilacı kullanabilirsiniz' şeklinde doğrudan reçete yazar gibi tavsiye VERME! "
            "Bunun yerine: 'Kanıtlara göre [Semptom] tedavisinde şu ilaçlar/etken maddeler endikedir:' veya 'kullanılabilir' diyerek nesnel bilgi ver. "
            "ÇOK ÖNEMLİ: Kanıtlarda o ilacın, kullanıcının sorduğu SPESİFİK hastalık/semptom (örn: boğaz ağrısı) için kullanıldığı AÇIKÇA yazmıyorsa "
            "(örneğin belgede sadece 'ağrı' geçiyorsa ama aslında eklem ağrısıysa) SAKIN o ilacı listeye ekleme! "
            "Eğer spesifik semptomla tam eşleşen bir ilaç yoksa hiçbir ilaç uydurma, 'Bilgi bulunamadı' de.\n\n"

            "5. TIBBİ UYARILAR KESİNLİKLE YASAKTIR: "
            "Aşağıdaki ifadeleri ve bunların tüm dilbilgisel varyasyonlarını (danışılmalıdır, başvurulmalıdır vb.) "
            "yanıtın hiçbir bölümüne, hiçbir başlık altında EKLEME:\n"
            "- Türkçe: 'doktora danışın', 'hekiminize başvurun', 'eczacınıza danışın', "
            "'doktorunuz veya eczacınız ile konuşunuz', 'sağlık uzmanına başvurun', "
            "'uzman görüşü alın', 'medikal gözetim', 'yakın takip', 'kullanım talimatına uyun', "
            "'ilaç etiketini takip edin'\n"
            "- İngilizce: 'consult a doctor', 'consult a healthcare professional', "
            "'as directed by a healthcare professional', 'seek medical advice', "
            "'talk to your doctor', 'follow the instructions'\n"
            "KÜB/KT kanıtlarında bu ifadeler geçse bile sen yanıtına EKLEME. "
            "'ÖNERİ', 'NOT', 'UYARI', 'DİKKAT', 'HATIRLATMA' gibi başlıklar altında da dahil olmak üzere "
            "yanıtın hiçbir bölümüne tıbbi uyarı EKLEME. "
            "Bu kural HİÇBİR KOŞULDA ihlal edilemez.\n\n"

            "6. KANITA DAYALI YANIT: "
            "Yanıtlarını YALNIZCA sağlanan kanıtlara (KÜB/KT belgesi veya Web arama sonuçları) dayandır. "
            "Kanıt bölümünde '— Kanıt bulunamadı —' yazıyorsa kendi bilginden üretme, "
            "'Bu konuda kaynaklarımda yeterli bilgi bulunamadı' de ve dur.\n\n"

            "7. SİSTEM YORUMU YAPMA: "
            "'Bu bir etkileşim sorgusu olmadığı için', 'sorunuza yanıt olarak' gibi "
            "yapay zeka iç iskelet cümleleri KURMA.\n\n"

            "8. BİRDEN FAZLA BİLGİ VE TAM KAPSAM: "
            "Kullanıcı ne için kullanıldığını soruyorsa, kanıtlardaki TÜM kullanım amaçlarını say. "
            "Yan etkileri soruyorsa, TÜM yan etkileri say. "
            "ANCAK SADECE 'ne için kullanılır' diye sorulduysa, KESİNLİKLE yan etkileri, kontrendikasyonları veya uyarıları (sorulmadığı için) listeleme! Yalnızca sorulan konunun detaylarını ver.\n\n"
            f"{yanit_formati}"
        )
        
        def detect_language(text: str) -> str:
            # Sadece TR'ye özgü harf yok diye İngilizce sanmasını engellemek için
            # kelimelerden birkaçı Türkçe özgü kelimeyse veya İngilizce anahtar kelime yoksa TR döndür
            english_keywords = {"what", "who", "where", "when", "why", "how", "is", "are", "do", "does", "invented", "side effects", "tell me"}
            words = set(text.lower().split())
            if words.intersection(english_keywords):
                return "English"
            return "Türkçe"

        detected_lang = detect_language(question)

        ilac_str = out.get('ilac_adi', '')
        if not ilac_str:
            global_hallucination = ""
        else:
            global_hallucination = (
                f"⚠️ GENEL HALÜSİNASYON KURALI: 1) Eğer kanıtlarda kullanıcının sorduğu ilaç ({ilac_str}) VEYA etken maddesi AÇIKÇA GEÇMİYORSA ve bambaşka bir marka/ilaçtan bahsediliyorsa, o alakasız ilacın bilgilerini '{ilac_str}' imiş gibi UYDURMA! Doğrudan 'Elimdeki kaynaklarda {ilac_str} hakkında bilgi bulunamadı.' de.\n"
                f"2) Kullanıcı 'ne için kullanılır' diye sorduğunda kanıtlarda AÇIK BİR KULLANIM ALANI (endikasyon) YOKSA ve SADECE ilacın kontrendikasyonları/uyarıları geçiyorsa, SAKIN bunları kullanım alanıymış gibi ters çevirip UYDURMA! Sadece 'kanıt bulunamadı' de. Ancak kanıtlarda '... tedavisinde kullanılır', '... endikedir' gibi gerçek kullanım alanları Varsa ONLARI LİSTELE.\n"
            )
            
        hallucination_rule = global_hallucination
        if query_type == "interaction":
            if not out.get('etkilesen_madde'):
                hallucination_rule += (
                    f"⚠️ GENEL ETKİLEŞİM KURALI: Kullanıcı '{out.get('ilac_adi', '')}' ilacının 'diğer/farklı' ilaçlarla genel etkileşimlerini soruyor. "
                    f"Geçmiş sohbetlerde geçen ALAKASIZ ilaç isimlerini (örn. sohbetin en başında sorulmuş tamamen başka bir ilacı) KESİNLİKLE bu ilacın etkileşimiymiş gibi yanıtına yazma! "
                    f"SADECE aşağıdaki RAG kanıtlarında '{out.get('ilac_adi', '')}' veya etken maddesi ile birlikte kullanıldığında etkileşime veya uyarıya neden olan diğer ajanları listele.\n\n"
                )
            else:
                hallucination_rule += (
                    f"⚠️ SPESİFİK ETKİLEŞİM HALÜSİNASYON KURALI: Kullanıcının SON SORUSUNDA açıkça belirttiği ilaçlar ({out.get('ilac_adi', '')} ve {out.get('etkilesen_madde', '')}) ile RAG kanıtlarındaki ilaç isimlerini EŞLEŞTİR. "
                    f"1) Eğer kanıtlarda bambaşka iki ilacın (örn: kalsiyum ve tiyazid) etkileşiminden bahsediliyorsa, bunu asla kullanıcının sorduğu ilaçların (örn: kalsiyum ve levotiroksin) etkileşimiymiş gibi yansıtma!\n"
                    f"2) Bilmediğin veya kanıtta CÜMLE İÇİNDE birbirine bağlanmayan ilaçlar için sahte mekanizmalar uydurma. Eğer Kural Motoru RİSK: HIGH diyorsa fakat RAG kanıtları ikisinin BİRBİRİYLE olan etkileşim sebebini İÇERMİYORSA (ancak kanıtta başka ilaçların etkileşimi geçiyorsa), sadece risk seviyesini belirt ve aynen şunu söyle: 'Kanıtlarda bu kombinasyonun neden riskli olduğuna dair detaylı açıklama yer almamaktadır, ancak diğer ilaçlar ile olan farklı etkileşim uyarıları bulunmaktadır.' (ANCAK kullanıcı açıkça 'farklı uyarılar neler', 'diğerleri neler', 'nelerdir' gibi takip soruları sorduysa bu kuralı es geçip, kanıtlarda geçen diğer etkileşimleri ve uyarıları listele.)\n\n"
                )

        evidence_label = "🌐 WEB ARAMA SONUÇLARI:" if (evidence_text or "").startswith("[WEB KAYNAĞI") else "📗 KÜB/KT KANITLARI (Vektör DB):"

        user_prompt = (
            f"⚠️ DİL KURALI: Yanıtını YALNIZCA {detected_lang} dilinde yaz. "
            f"Başka dil kullanma.\n\n"
            f"SORU: {question}\n\n"
            f"ÖNEMLİ: Kullanıcının sorusu '{question}' — "
            f"yanıtın YALNIZCA bu soruyu cevaplar nitelikte olmalı. \n\n"
            f"{hallucination_rule}\n"
            f"Kanıtlar içinde soruyla ilgili olmayan bölümleri (yan etkiler vb.) sorulmadıkça yanıta dahil etme.\n\n"
            f"TESPİT EDİLEN İLAÇ/MADDE/SEMPTOM BİLGİLERİ:\n"
            f"- İlaç Adı: {out.get('ilac_adi', '') or 'Belirtilmedi'}\n"
            f"- İlgili Madde/Semptom: {out.get('etkilesen_madde', '') or out.get('yan_etki', '') or 'Belirtilmedi'}\n\n"
            f"{risk_input}"
            f"{evidence_label}\n"
            f"{evidence_text if evidence_text else '— Kanıt bulunamadı —'}\n\n"
            "Sadece sorulan konuyu yanıtla. Selamlama ve doktora danışın uyarısı KULLANMA."
        )

        msg_list = [SystemMessage(content=system_prompt)]
        
        # Sadece son 4 mesajı (2 soru-cevap) bağlam olarak gönder, LLM'in kafası karışmasın
        recent_messages = out.get("messages", [])[-4:]
        for m in recent_messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                msg_list.append(HumanMessage(content=content))
            elif role == "assistant":
                msg_list.append(AIMessage(content=content))

        msg_list.append(HumanMessage(content=user_prompt))

        msg = model.invoke(
            msg_list,
            config={"callbacks": _get_observability_callbacks()}
        )

        out["yanit"] = getattr(msg, "content", str(msg))
        return out
    except RuntimeError as e:
        # If Groq isn't configured, return deterministic info + evidence (no LLM).
        out["yanit"] = (
            "LLM yapılandırılamadı. Kural motoru sonucu ve RAG kanıtlarıyla sınırlı yanıt:\n\n"
            f"- Soru: {question}\n"
            f"- Sorgu türü: {query_type}\n"
            f"- İlaç: {out.get('ilac_adi','')}\n"
            f"- Risk seviyesi: {risk_seviyesi}\n"
            f"RAG kanıtları:\n{evidence_text if evidence_text else '—'}\n\n"
            f"Hata: {e}"
        )
        return out


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("categorize_query", categorize_query)
    g.add_node("check_rules", check_rules)
    g.add_node("get_explanation", get_explanation)

    g.set_entry_point("categorize_query")
    g.add_edge("categorize_query", "check_rules")
    g.add_edge("check_rules", "get_explanation")
    g.add_edge("get_explanation", END)

    return g.compile()


def run_agent(question: str, messages: list = None) -> AgentState:
    graph = build_graph()
    return graph.invoke({"question": question, "messages": messages or []})

def export_graph_mermaid(output_path: str = "graph.md") -> str:
    """LangGraph akışını Mermaid formatında dışa aktar."""
    graph = build_graph()
    mermaid_str = graph.get_graph().draw_mermaid()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"```mermaid\n{mermaid_str}\n```")
    print(f"Mermaid grafiği {output_path} dosyasına yazıldı.")
    return mermaid_str


if __name__ == "__main__":
    # Simple manual test:
    print(run_agent("Warfarin ile greyfurt suyu etkileşimi nedir?"))


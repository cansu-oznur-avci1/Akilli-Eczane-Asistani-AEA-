from __future__ import annotations

import re
from typing import List, Literal, TypedDict

from langgraph.graph import END, StateGraph

from engine.rule_engine import QueryType, RiskLevel, RuleEngine
from vector_db.retrieval import format_evidence, retrieve


class AgentState(TypedDict, total=False):
    question: str
    query_type: Literal["interaction", "side_effect", "general_info", "unknown"]

    ilac_adi: str
    etkilesen_madde: str
    yan_etki: str

    risk_seviyesi: str
    kaynaklar: List[str]

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
    "doz",
    "endikasyon",
    "kontrendikasyon",
    "uyarı",
    "etken madde",
    "farmakoloji",
    "özellik",
    "genel bilgi",
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
    return "unknown"


def _extract_interaction_entities(q: str) -> tuple[str, str]:
    q_norm = q.replace("\n", " ").strip()
    # Common patterns:
    # "X ile Y etkileşimi" / "X ve Y etkileşimi" / "X ile Y birlikte"
    patterns = [
        r"(?P<a>.+?)\s+(?:ile|ve)\s+(?P<b>.+?)\s+etkile\w*",
        r"(?P<a>.+?)\s+(?:ile|ve)\s+(?P<b>.+?)\s+(?:birlikte|aynı anda|eş zamanlı)",
        r"(?P<a>.+?)\s+(?:ile|birlikte)\s+(?P<b>.+)$",
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

    # last-resort: split on " ile "
    if " ile " in q_norm.casefold():
        parts = re.split(r"\bile\b", q_norm, flags=re.IGNORECASE)
        if len(parts) >= 2:
            return _normalize_entity(parts[0]), _normalize_entity(parts[1])

    return "", ""


def _extract_side_effect_entities(q: str) -> tuple[str, str]:
    q_norm = q.replace("\n", " ").strip()
    # "X’in yan etkisi", "X'in yan etkileri"
    m = re.search(
        r"(?P<ilac>.+?)\s*(?:’|')?in\s+yan\s+etk\w*",
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


def categorize_query(state: AgentState) -> AgentState:
    q = state.get("question", "") or ""
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

    # general_info/unknown: extraction is intentionally best-effort; retrieval uses the whole question
    return out


def check_rules(state: AgentState) -> AgentState:
    out: AgentState = dict(state)
    out["kaynaklar"] = out.get("kaynaklar", [])

    query_type = out.get("query_type", "unknown")
    engine = RuleEngine()

    if query_type == "interaction":
        ilac = out.get("ilac_adi", "")
        madde = out.get("etkilesen_madde", "")
        if ilac and madde:
            rec = engine.lookup_typed(ilac, madde, QueryType.INTERACTION)
            risk = rec.risk_seviyesi if rec else RiskLevel.UNKNOWN
            out["risk_seviyesi"] = risk.value
            out["kaynaklar"] = [rec.kaynak] if rec and rec.kaynak else []
        else:
            out["risk_seviyesi"] = RiskLevel.UNKNOWN.value
    else:
        out["risk_seviyesi"] = RiskLevel.UNKNOWN.value

    return out


def get_explanation(state: AgentState) -> AgentState:
    out: AgentState = dict(state)
    question = out.get("question", "") or ""
    query_type = out.get("query_type", "unknown")

    risk_seviyesi = out.get("risk_seviyesi", RiskLevel.UNKNOWN.value)

    # Decide whether RAG is needed.
    # - interaction: fetch evidence for HIGH/LOW/UNKNOWN, skip for NONE
    # - side_effect/general_info: always fetch
    do_rag = True
    if query_type == "interaction" and risk_seviyesi == RiskLevel.NONE.value:
        do_rag = False

    # Deterministic response for NONE (no LLM, no RAG evidence needed).
    if query_type == "interaction" and not do_rag:
        kaynaklar = out.get("kaynaklar", [])
        out["yanit"] = (
            f"Kural motoru risk seviyesi: {risk_seviyesi}\n"
            f"Kaynak(lar): {', '.join(kaynaklar) if kaynaklar else '—'}"
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
            retrieval_query = f"{out.get('ilac_adi','')} {out.get('yan_etki','')}".strip() or question
        else:
            retrieval_query = question

        chunks = retrieve(retrieval_query, k=4)
        evidence_chunks = [c.text for c in chunks]
        evidence_text = format_evidence(chunks, max_chars=4200)
    else:
        evidence_text = ""

    out["rag_chunks"] = evidence_chunks
    out["rag_ozet"] = evidence_text

    # Safety: RAG kanıtı yoksa LLM'e hiç çağrı yapmıyoruz.
    # Böylece uydurma içerik üretme riskini düşürürüz.
    if do_rag and not (evidence_text or "").strip():
        kaynaklar = out.get("kaynaklar", [])
        risk = out.get("risk_seviyesi", RiskLevel.UNKNOWN.value)

        if query_type == "interaction":
            out["yanit"] = (
                f"Kural motoru risk seviyesi: {risk}\n"
                f"Kaynak(lar): {', '.join(kaynaklar) if kaynaklar else '—'}\n\n"
                "İlgili KÜB/KT kanıt parçaları bulunamadı. "
                "RAG için önce `vector_db/ingest_data.py` ile PDF’leri ingest edin."
            )
        else:
            out["yanit"] = (
                "İlgili KÜB/KT kanıt parçaları bulunamadı. "
                "RAG için önce `vector_db/ingest_data.py` ile PDF’leri ingest edin."
            )
        return out

    # LLM part (Groq) - must not change risk decision; only summarize evidence in plain Turkish.
    from backend.llm.groq_client import build_chat_model

    try:
        model = build_chat_model()
        system_prompt = (
            "Sen bir Akıllı Eczacı Asistanısın. "
            "Görevlerin: kullanıcının sorusuna, KÜB/KT metinlerinden (RAG) ve Kural Motoru risk sonucundan "
            "hareketle, açıklayıcı ve sade bir dilde yanıt üretmek.\n"
            "Kurallar:\n"
            "1) Tıbbi teşhis koyma ve kesin tedavi talimatı verme.\n"
            "2) Risk seviyesi (HIGH/LOW/NONE/UNKNOWN) sadece Kural Motoru tarafından belirlenir; "
            "LLM bunu değiştirmeyecektir.\n"
            "3) RAG kanıtı yoksa uydurma yapma; eldeki bilgilerle sınırlı kal.\n"
        )

        user_prompt = (
            f"Soru: {question}\n\n"
            f"Sorgu türü: {query_type}\n"
            f"İlaç: {out.get('ilac_adi','')}\n"
            f"Etkileşen madde / yan etki: {out.get('etkilesen_madde','') or out.get('yan_etki','')}\n"
            f"Kural motoru risk seviyesi: {risk_seviyesi}\n\n"
            "RAG'den gelen KÜB/KT kanıtları:\n"
            f"{evidence_text if evidence_text else '—'}\n\n"
            "Yanıt formatı:\n"
            "- Önce risk seviyesini (varsa) ve kısa gerekçesini belirt.\n"
            "- Ardından KÜB/KT kanıtlarına dayalı, sade ve anlaşılır açıklama yap.\n"
            "- Tehlikeli risk ise kullanıcıyı ilgili sağlık profesyoneline başvuruya yönlendiren pratik bilgi ver.\n"
            "Metni Türkçe yaz ve tanısal ifadelerden kaçın."
        )

        msg = model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
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
            f"- Kaynak(lar): {', '.join(out.get('kaynaklar', [])) if out.get('kaynaklar') else '—'}\n\n"
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


def run_agent(question: str) -> AgentState:
    graph = build_graph()
    return graph.invoke({"question": question})


if __name__ == "__main__":
    # Simple manual test:
    print(run_agent("Warfarin ile greyfurt suyu etkileşimi nedir?"))


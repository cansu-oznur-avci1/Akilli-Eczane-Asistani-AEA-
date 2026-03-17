from __future__ import annotations

from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from engine.rule_engine import RiskLevel, RuleEngine


class AEAState(TypedDict, total=False):
    ilac_adi: str
    etkilesen_madde: str
    risk_seviyesi: str
    kaynaklar: List[str]
    rag_ozet: str
    yanit: str


def node_input(state: AEAState) -> AEAState:
    # Input node exists mainly for structure/validation later.
    return state


def node_rule_engine(state: AEAState) -> AEAState:
    engine = RuleEngine()
    ilac = state.get("ilac_adi", "")
    madde = state.get("etkilesen_madde", "")
    rec = engine.lookup(ilac, madde)
    risk = rec.risk_seviyesi if rec else RiskLevel.UNKNOWN

    out: AEAState = dict(state)
    out["risk_seviyesi"] = risk.value
    out["kaynaklar"] = [rec.kaynak] if rec and rec.kaynak else []
    return out


def node_rag(state: AEAState) -> AEAState:
    """
    Explanatory-only retrieval. Not a decision-maker.

    Placeholder: to be implemented with embeddings + vector store.
    """

    out: AEAState = dict(state)
    risk = RiskLevel(out.get("risk_seviyesi", RiskLevel.UNKNOWN.value))

    if risk in (RiskLevel.NONE,):
        out["rag_ozet"] = ""
        return out

    # TODO: Replace with real vector DB retrieval and summarization.
    out["rag_ozet"] = (
        "Bu bölüm RAG (retrieval) için iskelettir. "
        "Kural motoru risk tespit ettiğinde veya belirsiz olduğunda, "
        "bilimsel kaynaklardan kısa bir açıklama çekilecek."
    )
    return out


def node_llm(state: AEAState) -> AEAState:
    """
    Generates plain-language explanation without diagnosis.
    Must respect RuleEngine decision; never upgrades/downgrades risk.
    """

    from backend.llm.groq_client import build_chat_model

    out: AEAState = dict(state)

    ilac = out.get("ilac_adi", "")
    madde = out.get("etkilesen_madde", "")
    risk = out.get("risk_seviyesi", RiskLevel.UNKNOWN.value)
    rag_ozet = out.get("rag_ozet", "")
    kaynaklar = out.get("kaynaklar", [])

    try:
        model = build_chat_model()
    except RuntimeError as e:
        out["yanit"] = (
            "LLM yapılandırması bulunamadı (ör. `GROQ_API_KEY`). "
            "Bu nedenle yalnızca kural motoru çıktısı döndürülüyor.\n\n"
            f"İlaç: {ilac}\n"
            f"Etkileşen madde/besin: {madde}\n"
            f"Kural motoru risk seviyesi: {risk}\n"
            f"Kaynak(lar): {', '.join(kaynaklar) if kaynaklar else '—'}\n\n"
            f"Hata: {e}"
        )
        return out

    system = (
        "Sen bir Akıllı Eczacı Asistanı çıktısı üretiyorsun. "
        "Tıbbi teşhis koyma. Kesin karar verme. "
        "Risk seviyesini sadece 'Kural Motoru' belirler; bunu değiştirme. "
        "Halk dilinde, kısa ve anlaşılır yaz. "
        "Acil/tehlikeli risklerde doktora/eczacıya yönlendir."
    )
    user = (
        f"İlaç: {ilac}\n"
        f"Etkileşen madde/besin: {madde}\n"
        f"Kural motoru risk seviyesi: {risk}\n"
        f"RAG özeti (bilimsel açıklama olabilir): {rag_ozet}\n"
        f"Kaynak(lar): {', '.join(kaynaklar) if kaynaklar else '—'}\n\n"
        "Kullanıcıya tek bir yanıt üret."
    )

    msg = model.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )

    out["yanit"] = getattr(msg, "content", str(msg))
    return out


def node_output(state: AEAState) -> AEAState:
    return state


def _need_rag(state: AEAState) -> str:
    risk = RiskLevel(state.get("risk_seviyesi", RiskLevel.UNKNOWN.value))
    return "rag" if risk in (RiskLevel.HIGH, RiskLevel.LOW, RiskLevel.UNKNOWN) else "llm"


def build_graph():
    g = StateGraph(AEAState)
    g.add_node("Input", node_input)
    g.add_node("RuleEngine", node_rule_engine)
    g.add_node("RAG", node_rag)
    g.add_node("LLM", node_llm)
    g.add_node("Output", node_output)

    g.set_entry_point("Input")
    g.add_edge("Input", "RuleEngine")
    g.add_conditional_edges("RuleEngine", _need_rag, {"rag": "RAG", "llm": "LLM"})
    g.add_edge("RAG", "LLM")
    g.add_edge("LLM", "Output")
    g.add_edge("Output", END)

    return g.compile()


def run_once(ilac_adi: str, etkilesen_madde: str) -> AEAState:
    graph = build_graph()
    return graph.invoke({"ilac_adi": ilac_adi, "etkilesen_madde": etkilesen_madde})


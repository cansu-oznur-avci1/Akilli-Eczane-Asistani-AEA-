import os
import re
from typing import List, Literal, Any, Dict

from langgraph.graph import END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_tavily import TavilySearch

from engine.rule_engine import QueryType, RiskLevel, RuleEngine, check_drug_risk_tool
from vector_db.retrieval import format_evidence, retrieve
from backend.llm.groq_client import build_chat_model
from schema import AgentState

_INTERACTION_WORDS = [
    "etkileşim", "etkilesim", "etkileşimi", "etkilesimi", "birlikte", "aynı anda",
    "eş zamanlı", "birlikte al", "ilaç-ilaç", "ilaç-besin",
    "tüketebilir", "tüketilir", "alınabilir", "kullanılabilir mi", "beraber", "kombine",
]

_SIDE_EFFECT_WORDS = ["yan etki", "yanetki", "yan etkisi", "zarar", "advers", "ne olur"]

_GENERAL_WORDS = [
    "nedir", "kullanım", "kullanımı", "kullanılır", "ne için", "niçin", "doz",
    "endikasyon", "kontrendikasyon", "uyarı", "etken madde", "farmakoloji",
    "özellik", "genel bilgi", "ne işe", "nasıl kullan", "dozaj", "muadil",
    "eşdeğer", "prospektüs", "ne zaman", "kimler kullan", "içeriği", "içerir",
    "kullanılabilir", "hakkında bilgi", "bilgi ver",
]

def _normalize_entity(s: str) -> str:
    s = " ".join((s or "").strip().split())
    s = s.strip(".,;:!?()[]{}\"'“”‘’")
    return s

def _categorize_query_text(q: str) -> str:
    q = q.casefold()
    if any(w in q for w in _SIDE_EFFECT_WORDS): return "side_effect"
    if any(w in q for w in _INTERACTION_WORDS): return "interaction"
    if any(w in q for w in _GENERAL_WORDS): return "general_info"
    return "general_info"

def _extract_interaction_entities(q: str) -> tuple[str, str]:
    q_norm = q.replace("\n", " ").strip()
    m_sonda_ile = re.search(r"^(?P<a>\S+)\s+(?P<b>.+?)\s+ile\s+(?:kullanıl|tüketil|alın|birlikte)", q_norm, flags=re.IGNORECASE)
    if m_sonda_ile:
        a, b = _normalize_entity(m_sonda_ile.group("a")), _normalize_entity(m_sonda_ile.group("b"))
        if a and b and len(a) >= 2 and len(b) >= 2: return a, b
    patterns = [
        r"(?P<a>.+?)\s+(?:ile|ve)\s+(?P<b>.+?)\s+etkile\w*",
        r"(?P<a>.+?)\s+(?:ile|ve)\s+(?P<b>.+?)\s+(?:birlikte|aynı anda|eş zamanlı)",
        r"(?P<a>.+?)\s+(?:ile|birlikte)\s+(?P<b>.+)$",
        r"(?P<a>[A-Za-zÇçĞğİıÖöŞşÜüa-z]+(?:\s+[A-Za-zÇçĞğİıÖöŞşÜüa-z]+)*?)\s+(?P<b>.+?)\s+ile\s+birlikte",
        r"(?P<a>.+?)\s+ile\s+(?P<b>.+?)\s+kullanıl",
    ]
    for pat in patterns:
        m = re.search(pat, q_norm, flags=re.IGNORECASE)
        if m:
            a, b = _normalize_entity(m.group("a")), _normalize_entity(m.group("b"))
            if a and b and len(a) <= 80 and len(b) <= 80: return a, b
    if " ile " in q_norm.casefold():
        m_once = re.search(r"^(?P<a>.+?)\s+ile\s+(?P<b>.+?)\s+birlikte", q_norm, flags=re.IGNORECASE)
        if m_once: return _normalize_entity(m_once.group("a")), _normalize_entity(m_once.group("b"))
        m_std = re.search(r"^(?P<a>.+?)\s+ile\s+(?P<b>.+?)(?:\s+birlikte|\s+kullanıl|\s+tüket|\s+alın|\s*$)", q_norm, flags=re.IGNORECASE)
        if m_std: return _normalize_entity(m_std.group("a")), _normalize_entity(m_std.group("b"))
    return "", ""

def _extract_side_effect_entities(q: str) -> tuple[str, str]:
    q_norm = q.replace("\n", " ").strip()
    m = re.search(r"(?P<ilac>.+?)\s*(?:’|')?(?:in|ın|un|ün|nin|nın|nun|nün)?\s+yan\s+etk\w*", q_norm, flags=re.IGNORECASE)
    if m: return _normalize_entity(m.group("ilac")), ""
    if " ile " in q_norm.casefold():
        parts = re.split(r"\bile\b", q_norm, flags=re.IGNORECASE)
        return _normalize_entity(parts[0]), ""
    return "", ""

def _extract_general_entities(q: str) -> str:
    q_norm = q.replace("\n", " ").strip()
    m = re.search(r"(?P<ilac>.+?)\s+(?:ne için|ne işe|nedir|nasıl|niçin|kullanım|kullanılır|doz|endikasyon|kontrendikasyon|yan etki)", q_norm, flags=re.IGNORECASE)
    if m: return _normalize_entity(m.group("ilac"))
    
    # Eğer ilk kelimeler ilaç adıysa ve cümlenin devamı varsa (örn: "aferin forte hakkında bilgi ver")
    words = q_norm.split()
    if len(words) <= 3 and not any(w in q_norm.lower() for w in ["nedir", "için"]): return _normalize_entity(q_norm)
    if len(words) > 0 and len(words[0]) > 2:
        return _normalize_entity(" ".join(words[:2]))
    return ""

async def categorize_query(state: AgentState) -> dict:
    state_dict = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    q = state_dict.get("question", "") or ""
    messages = list(state_dict.get("messages", []))
    query_type = state_dict.get("query_type") or _categorize_query_text(q)

    out = dict(state_dict)
    out["query_type"] = query_type

    if query_type == "interaction" and not (out.get("ilac_adi") and out.get("etkilesen_madde")):
        ilac, madde = _extract_interaction_entities(q)
        out["ilac_adi"] = ilac
        out["etkilesen_madde"] = madde

    elif query_type == "side_effect" and not out.get("ilac_adi"):
        ilac, yan_etki = _extract_side_effect_entities(q)
        out["ilac_adi"] = ilac
        out["yan_etki"] = yan_etki

    elif query_type == "general_info" and not out.get("ilac_adi"):
        ilac = _extract_general_entities(q)
        out["ilac_adi"] = ilac

    def needs_history_fallback(q_text, ext_ilac):
        if ext_ilac:
            # Check if ext_ilac is ONLY a cue word / pronoun (e.g., just "bu", "o", "peki", "diğer", "bunun")
            cues = {"peki", "başka", "bunun", "bununla", "ilaç", "ilaçla", "onunla", "diğer", "bu", "şu", "o", "vitamin", "onu"}
            if ext_ilac.lower().strip() in cues:
                return True
            return False
            
        if not ext_ilac:
            cues = {"yan etki", "peki", "başka", "birlikte", "bunun", "etkileşim", "onu", "doz", "kullanım", "kullanılır", "nedir", "için", "nasıl"}
            if any(c in q_text.lower() for c in cues) or "ile " in q_text.lower():
                symptom_cues = ["ağrı", "akıntı", "ateş", "öksürük", "mide", "bulantı", "hastalık", "yanıyor", "var", "oldum", "kötü"]
                if any(s in q_text.lower() for s in symptom_cues): return False
                return True
            if len(q_text.split()) <= 3: return True
        return False

    # Remove non-entity filler words from the extracted ilac_adi before assigning
    raw_ilac = out.get("ilac_adi", "")
    if raw_ilac:
        filler_words = ["peki", "ama", "fakat", "haliyle", "şimdi", "o zaman", "bu durumda", "pki"]
        clean_ilac = raw_ilac.casefold()
        for fw in filler_words:
            if clean_ilac.startswith(fw + " "):
                clean_ilac = clean_ilac[len(fw)+1:].strip()
        out["ilac_adi"] = raw_ilac[-len(clean_ilac):] if len(clean_ilac) > 0 else raw_ilac

    if needs_history_fallback(q, out.get("ilac_adi")):
        prev_ilac = ""
        prev_madde = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                prev_q = m.get("content", "")
                ptype = _categorize_query_text(prev_q)
                pm = ""
                if ptype == "interaction":
                    pi, pm = _extract_interaction_entities(prev_q)
                elif ptype == "side_effect":
                    pi, _ = _extract_side_effect_entities(prev_q)
                else:
                    pi = _extract_general_entities(prev_q)
                if pi:
                    prev_ilac = pi
                    prev_madde = pm
                    break
        
        if prev_ilac: out["ilac_adi"] = prev_ilac
        if prev_madde:
            if not any(w in q.lower() for w in ["diğer", "başka", "tüm", "genel", "farklı"]):
                out["etkilesen_madde"] = prev_madde

    return out

tavily_search_tool = TavilySearch(
    max_results=3,
    include_domains=['scholar.google.com', 'drugs.com', 'pubmed.ncbi.nlm.nih.gov', 'dergipark.org.tr'],
    name="tavily_search_results_json"
)

async def rag_node(state: AgentState) -> dict:
    state_dict = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    out = dict(state_dict)
    question = out.get("question", "") or ""
    query_type = out.get("query_type", "unknown")

    if not out.get("rag_ozet") and "rag_ozet" in out:
        do_rag = True
        evidence_text = ""
        evidence_chunks = []
        if query_type == "interaction":
            retrieval_query = f"{out.get('ilac_adi','')} {out.get('etkilesen_madde','')}".strip() or question
        elif query_type == "side_effect":
            yan = out.get('yan_etki', '').strip() or "yan etki"
            retrieval_query = f"{out.get('ilac_adi', '').strip()} {yan}".strip()
        else:
            ilac = out.get("ilac_adi", "").strip()
            retrieval_query = f"{ilac} {question}" if ilac and ilac.lower() not in question.lower() else question

        import os
        k_value = int(os.getenv("AEA_RAG_K", "4"))
        chunks = retrieve(retrieval_query, k=k_value)
        evidence_chunks = [c.text for c in chunks]
        evidence_text = format_evidence(chunks, max_chars=4200)

        # Include source paths in the check to prevent false rejection of chunks from correctly named PDFs
        source_paths = " ".join([getattr(c, "source_path", "") for c in chunks]).casefold()

        ilac_check = (out.get("ilac_adi") or "").strip().casefold()
        if ilac_check and evidence_text and len(ilac_check) >= 3:
            combined_text = (" ".join(evidence_chunks) + " " + source_paths).casefold()
            # Split ilac name and filter out filler words so we only check core drug name
            ilac_tokens = [t for t in ilac_check.split() if len(t) >= 3 and t not in ["için", "nedir", "nasıl", "peki", "başka", "olan", "ile"]]
            
            if ilac_tokens:
                # If ALL of the core drug name tokens don't appear (either in text or filename metadata), wipe the chunk.
                # This prevents "Aferin Sinüs" query from returning Nasal Spray PDFs just because "sinüs" matched.
                found_all = all(tok in combined_text for tok in ilac_tokens)
                if not found_all:
                    evidence_text = ""
                    evidence_chunks = []

        out["rag_chunks"] = evidence_chunks
        out["rag_ozet"] = evidence_text

    return out

async def rule_engine_node(state: AgentState) -> dict:
    state_dict = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    out = dict(state_dict)
    
    ilac = out.get("ilac_adi", "")
    madde = out.get("etkilesen_madde", "")
    if ilac and madde:
        res = check_drug_risk_tool.invoke({"ilac_adi": ilac, "hedef": madde, "query_type": "interaction"})
        try:
            import json
            r = json.loads(res)
            out["risk_level"] = r.get("risk_seviyesi", "")
            out["rag_ozet"] = f"[KURAL MOTORU SONUCU]\nRisk Seviyesi: {r.get('risk_seviyesi', '')}\nAçıklama: {r.get('aciklama', '')}"
        except json.JSONDecodeError:
            # If not JSON, parse the string "Kural Motoru Risk Seviyesi: HIGH"
            risk_val = "Bilinmiyor"
            if "Risk Seviyesi:" in res:
                risk_val = res.split("Risk Seviyesi:")[-1].strip()
            out["risk_level"] = risk_val
            out["rag_ozet"] = f"[KURAL MOTORU SONUCU]\nRisk Seviyesi: {risk_val}\nAçıklama: Kurallara dayalı risk analizi tamamlandı."
    return out

async def tavily_node(state: AgentState) -> dict:
    state_dict = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    out = dict(state_dict)
    
    q = out.get("question", "")
    if q and os.getenv("TAVILY_API_KEY"):
        try:
            res = tavily_search_tool.invoke({"query": q})
            out["raw_research_data"] = str(res)
        except Exception as e:
            out["raw_research_data"] = f"Arama hatası: {e}"
    return out

async def agent_node(state: AgentState) -> dict:
    state_dict = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    out = dict(state_dict)
    question = out.get("question", "") or ""
    messages = list(out.get("messages", []))

    KIMLIK_KELIMELERI = ["kimsin", "kim yaptı", "kim icat", "kim geliştirdi", "kim kurdu", "seni kim", "ne yaparsın", "kendin tanıt", "hakkında bilgi ver", "who are you", "what are you"]
    if any(k in question.casefold() for k in KIMLIK_KELIMELERI):
        if not messages or not isinstance(messages[-1], AIMessage):
            out["yanit"] = "Ben bilgisayar mühendisliği öğrencileri (Cansu Öznur AVCI, Asya Mina ATİK, Elifnur ŞİMŞEK) tarafından Prof. Dr. Ramazan Katırcı danışmanlığında geliştirilen Akıllı Eczacı Asistanıyım (AEA)."
            messages.append(AIMessage(content=out["yanit"]))
            out["messages"] = messages
        return out

    lang = out.get("user_info", {}).get("lang", "tr")
    
    if lang == "en":
        system_prompt = (
            "YOUR TASK: You are a highly professional, knowledgeable, empathetic AI named 'Smart Pharmacist Assistant (SPA)'.\n"
            "You must respond with the depth of an expert pharmacist, using a detailed, informative, and patient-friendly tone.\n\n"
            "STRICT RULES:\n"
            "1. NO MEDICAL WARNINGS: NEVER use phrases like 'Consult your doctor', 'Seek medical advice', 'not medical advice', etc. Provide direct, clear information.\n"
            "2. DETAILED & ACCURATE: Do not give short answers. Explain the mechanism, usage, and details scientifically yet simply.\n"
            "3. USE EVIDENCE: Use provided CURRENT SOURCES. If no info, use your expert pharmacology knowledge. Do not say 'Not found in sources', just answer naturally.\n"
            "4. MAINTAIN CONTEXT: Consider chat history but do not mix old drugs with new ones.\n"
            "5. STATE RISK LEVEL: If a Risk Level is provided, state it clearly and explain it.\n"
            "IMPORTANT: YOU MUST RESPOND IN ENGLISH."
        )
        evidence_label = "🌐 SOURCES:"
    else:
        system_prompt = (
            "GÖREVİN: Sen, 'Akıllı Eczacı Asistanı (AEA)' adında son derece profesyonel, bilgili, empatik ve aydınlatıcı bir yapay zekasın.\n"
            "Sıradan bir asistan gibi değil, alanında uzman, tecrübeli bir eczacı kimliğiyle, detaylı ve özel bir üslupla yanıt vermelisin.\n\n"
            "KATÎ KURALLAR:\n"
            "1. TIBBİ UYARI YASAK: Yanıtlarında KESİNLİKLE 'doktorunuza danışın', 'eczacınıza danışın', 'tıbbi tavsiye değildir', 'doktora başvurun', 'hekiminize danışın', 'sağlık kuruluşuna gidin' gibi ifadeler KULLANMA. Kullanıcıya doğrudan, net ve açıklayıcı bilgi sun.\n"
            "2. DETAYLI VE YAPIK (STRUCTURED) YANIT: Soruları asla kısa cümlelerle geçiştirme. İlacın etki mekanizmasını ve detaylarını alt başlıklar, madde imleri (bullet points) ve kalın yazılar (bold) kullanarak, son derece şık bir Markdown formatında sun.\n"
            "3. KANIT KULLANIMI: Sana sağlanan GÜNCEL KAYNAKLAR'ı öncelikli olarak kullan. Kaynaklarda yoksa genel tıbbi bilgiyle DOĞAL bir yanıt ver, 'kaynakta bulamadım' deme.\n"
            "4. BAĞLAMI KORU: Önceki sorulardaki ilaçların yan etkilerini yeni sorudaki ilaçlara KARIŞTIRMA.\n"
            "5. RİSK SEVİYESİ: Bağlamda bir 'Risk Seviyesi' varsa, bunu mutlaka belirt ve açıkla.\n"
            "ÖNEMLİ: Cevapların her zaman görsel olarak okunaklı, profesyonel paragraflar ve başlıklarla (Örn: ### Kullanım Amacı) yapılandırılmış olmalıdır."
        )
        evidence_label = "🌐 KAYNAKLAR:"
    actual_evidence = out.get("rag_ozet", "")
    if out.get("raw_research_data"):
        actual_evidence += "\n[TAVILY SONUÇLARI]\n" + out.get("raw_research_data", "")

    feedback_str = ""
    if out.get("reflexion_feedback"):
        if lang == "en":
            feedback_str = f"\n[CRITICAL CORRECTION NEEDED]: Your previous answer failed quality checks. Feedback: {out.get('reflexion_feedback')}\nPlease correct your answer.\n"
        else:
            feedback_str = f"\n[KRİTİK DÜZELTME GEREKİYOR]: Önceki cevabın kalite kontrolünden geçemedi. Geri bildirim: {out.get('reflexion_feedback')}\nLütfen cevabını buna göre düzelt.\n"

    user_prompt = (
        f"GÜNCEL SORU: {question}\n\n"
        f"ODAKLANILACAK İlaç: {out.get('ilac_adi', '')} | Madde: {out.get('etkilesen_madde', '')}\n"
        f"Risk Seviyesi:{out.get('risk_level', 'Bilinmiyor')}\n\n"
        f"GÜNCEL {evidence_label}\n"
        f"{actual_evidence if actual_evidence else ('No specific sources found. Use expert pharmacology knowledge.' if lang == 'en' else 'Sistem veritabanında bu ilaca dair özel belge bulunamadı. Lütfen eczacılık uzmanlığını kullanarak detaylı yanıtla.')}\n"
        f"{feedback_str}"
    )

    invoke_msgs = [SystemMessage(content=system_prompt)]
    
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                invoke_msgs.append(HumanMessage(content=content))
            elif role in ["assistant", "ai"]:
                invoke_msgs.append(AIMessage(content=content))
        elif hasattr(m, "type"):
            invoke_msgs.append(m)

    invoke_msgs.append(HumanMessage(content=user_prompt))

    model = build_chat_model()
    response = await model.ainvoke(invoke_msgs)

    messages.append(response)
    out["messages"] = messages
    
    raw_yanit = getattr(response, "content", "")
    
    # Post-processing: remove blocked phrases
    blocked_phrases = [
        "doktorunuza danışın", "eczacınıza danışın", "tıbbi tavsiye değildir", 
        "doktora başvurun", "hekiminize danışın", "sağlık kuruluşuna gidin",
        "consult your doctor", "seek medical advice", "not medical advice"
    ]
    import re
    for phrase in blocked_phrases:
        raw_yanit = re.sub(rf"(?i){phrase}[,.]?", "", raw_yanit)
        
    raw_yanit = re.sub(r"(?i)lütfen\s*\.", ".", raw_yanit)
    raw_yanit = re.sub(r"(?i)unutmayın ki\s*[,.]?", "", raw_yanit)
    
    out["yanit"] = raw_yanit.strip()

    return out

def route_after_categorize(state: AgentState) -> Literal["Kural_Motoru_Node", "Vektor_Veritabani_RAG"]:
    state_dict = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    if state_dict.get("query_type") == "interaction":
        return "Kural_Motoru_Node"
    return "Vektor_Veritabani_RAG"

def route_after_rag(state: AgentState) -> Literal["Web_Ara_Tavily_Node", "Asistan_LLM"]:
    state_dict = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    rag = str(state_dict.get("rag_ozet", ""))
    # Eğer sonuç 50 karakterden kısaysa (yani kanıt bulamadıysa) taviliye sap
    if len(rag) < 50 and os.getenv("TAVILY_API_KEY"):
        return "Web_Ara_Tavily_Node"
    return "Asistan_LLM"

async def reflexion_node(state: AgentState) -> dict:
    state_dict = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    out = dict(state_dict)
    
    yanit = out.get("yanit", "")
    risk_level = out.get("risk_level", "")
    rag_ozet = out.get("rag_ozet", "")
    loop_count = out.get("loop_count", 0)

    if not yanit or loop_count >= 2:
        out["reflexion_status"] = "PASS"
        return out

    model = build_chat_model().bind(response_format={"type": "json_object"})

    sys_msg = (
        "You are an AI Hallucination and Safety Checker. "
        "Review the AI's response against the provided Risk Level and Source Documents.\n"
        "1. Check if the response contradicts the Risk Level (e.g., saying 'safe' when risk is HIGH).\n"
        "2. Check if the response hallucinates medical facts not present in the Source Documents or general safe knowledge.\n"
        "Output ONLY JSON with two keys:\n"
        "- 'status': 'PASS' if acceptable, 'FAIL' if it needs correction.\n"
        "- 'feedback': If 'FAIL', explain what must be corrected. If 'PASS', leave empty."
    )
    
    user_msg = (
        f"Risk Level: {risk_level}\n"
        f"Sources: {rag_ozet}\n\n"
        f"AI Response to Evaluate: {yanit}"
    )

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json
        res = await model.ainvoke([SystemMessage(content=sys_msg), HumanMessage(content=user_msg)])
        result = json.loads(getattr(res, "content", "{}"))
        
        status = result.get("status", "PASS")
        out["reflexion_status"] = status
        out["reflexion_feedback"] = result.get("feedback", "")
        
        if status == "FAIL":
            out["loop_count"] = loop_count + 1
            
    except Exception as e:
        print(f"Reflexion Error: {e}")
        out["reflexion_status"] = "PASS"

    return out

def route_after_reflexion(state: AgentState) -> Literal["Asistan_LLM", "__end__"]:
    state_dict = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    status = state_dict.get("reflexion_status", "PASS")
    if status == "FAIL":
        return "Asistan_LLM"
    return "__end__"

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("Sorgu_Siniflandirma", categorize_query)
    g.add_node("Kural_Motoru_Node", rule_engine_node)
    g.add_node("Vektor_Veritabani_RAG", rag_node)
    g.add_node("Web_Ara_Tavily_Node", tavily_node)
    g.add_node("Asistan_LLM", agent_node)
    g.add_node("Halisunasyon_Kontrol_Node", reflexion_node)

    g.set_entry_point("Sorgu_Siniflandirma")
    
    g.add_conditional_edges("Sorgu_Siniflandirma", route_after_categorize)
    g.add_edge("Kural_Motoru_Node", "Vektor_Veritabani_RAG")
    
    g.add_conditional_edges("Vektor_Veritabani_RAG", route_after_rag)
    
    g.add_edge("Web_Ara_Tavily_Node", "Asistan_LLM")
    
    g.add_edge("Asistan_LLM", "Halisunasyon_Kontrol_Node")
    g.add_conditional_edges("Halisunasyon_Kontrol_Node", route_after_reflexion)

    return g.compile()

async def run_agent_async(question: str, messages: list = None) -> dict:
    graph = build_graph()
    initial_state = AgentState(question=question, messages=messages or [], user_info={}).model_dump()
    return await graph.ainvoke(initial_state)

def run_agent(question: str, messages: list = None) -> dict:
    import asyncio
    return asyncio.run(run_agent_async(question, messages))

def run_agent_stream(question: str, messages: list = None, lang: str = "tr"):
    """
    Synchronous generator that yields (node_name, state_dict) for Streamlit UI updates.
    Wraps the async graph execution in a thread to allow real-time synchronous yielding.
    """
    import threading
    import queue
    import asyncio
    
    graph = build_graph()
    initial_state = AgentState(question=question, messages=messages or [], user_info={"lang": lang}).model_dump()
    
    q = queue.Queue()
    
    async def _stream():
        try:
            async for output in graph.astream(initial_state):
                q.put(("output", output))
        except Exception as e:
            q.put(("error", e))
        finally:
            q.put(("done", None))
            
    def _run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_stream())
        loop.close()
        
    t = threading.Thread(target=_run_loop)
    t.start()
    
    while True:
        msg_type, data = q.get()
        if msg_type == "done":
            break
        elif msg_type == "error":
            raise data
        elif msg_type == "output":
            for node_name, state in data.items():
                yield node_name, state

def export_graph_mermaid(output_path: str = "AEA_flow.png") -> str:
    graph = build_graph()
    mermaid_str = graph.get_graph().draw_mermaid()
    try:
        from utils.visualizer import export_graph_mermaid_png
        export_graph_mermaid_png(mermaid_str, output_path)
    except ImportError as e:
        print(f"utils.visualizer modülü yüklenemedi: {e}")
        with open("AEA_flow.md", "w", encoding="utf-8") as f:
            f.write(f"```mermaid\\n{mermaid_str}\\n```")
    return mermaid_str

def generate_chat_title(question: str, lang: str = "tr") -> str:
    from backend.llm.groq_client import build_chat_model
    from langchain_core.messages import SystemMessage, HumanMessage
    model = build_chat_model()
    
    if lang == "en":
        sys_msg = "Generate a very short, concise title (max 4 words) for the user's question. Do not use quotes or punctuation."
    else:
        sys_msg = "Kullanıcının sorusu için en fazla 4 kelimelik çok kısa ve öz bir başlık üret. Tırnak işareti veya noktalama kullanma."
        
    try:
        response = model.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=question)
        ])
        title = getattr(response, "content", "").strip('"\'').strip()
        return title if title else (question[:30] + "...")
    except Exception:
        return question[:30] + "..."

if __name__ == "__main__":
    export_graph_mermaid()

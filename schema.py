from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    """
    Sistemin tüm süreç boyunca kullanacağı LangGraph state modeli.
    Kullanıcıdan gelen girdilerin doğrulanması (validation) ve ajanlar arası veri aktarımı için kullanılır.
    """
    question: str = Field(default="", description="Kullanıcının sorduğu soru metni.")
    messages: List[Any] = Field(default_factory=list, description="Mesajlaşma geçmişi (sohbet bağlamı).")
    user_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Kullanıcı metadataları (yaş, alerji, cinsiyet vb.)")
    
    query_type: Optional[Literal["interaction", "side_effect", "general_info", "unknown"]] = Field(
        default=None, description="Sorgu tipi sınıflandırması."
    )

    ilac_adi: Optional[str] = Field(default="", description="Tespit edilen ana ilaç/madde adı.")
    etkilesen_madde: Optional[str] = Field(default="", description="Tespit edilen ikincil etkileşen madde (etkileşim sorgularında).")
    yan_etki: Optional[str] = Field(default="", description="Tespit edilen yan etki semptomu.")

    risk_level: Optional[str] = Field(default="", description="Kural motorundan dönen risk seviyesi değerlendirmesi (eski adıyla risk_seviyesi).")

    rag_chunks: List[str] = Field(default_factory=list, description="Vektör veritabanından çekilen kanıt metinleri.")
    rag_ozet: Optional[str] = Field(default="", description="Bağlam veya Web araması sonucu oluşturulan özet.")
    raw_research_data: Optional[str] = Field(default="", description="Tavily API üzerinden asenkron kaydedilen araştırma verileri.")
    yanit: Optional[str] = Field(default="", description="LLM tarafından veya deterministik olarak üretilen son yanıt.")

    loop_count: int = Field(default=0, description="Halisünasyon kontrol döngüsü sayacı.")
    reflexion_status: Optional[str] = Field(default="", description="Reflexion node PASS/FAIL durumu.")
    reflexion_feedback: Optional[str] = Field(default="", description="Reflexion node tarafından verilen düzeltme geri bildirimi.")


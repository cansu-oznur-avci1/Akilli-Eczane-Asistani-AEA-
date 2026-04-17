from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# .env dosyasını modül yüklenirken bir kere oku.
# override=False: Admin panelinden sonradan set edilen os.environ değerlerini ezmez.
load_dotenv(override=False)  # İlk yükleme: mevcut env değerlerini korur


@dataclass(frozen=True)
class GroqConfig:
    api_key: str
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.2


def load_groq_config(dotenv_path: Optional[str] = None) -> GroqConfig:
    # dotenv_path verilirse o dosyayı da oku.
    # override=True: Dışarıdan verilen env değerleri (admin paneli os.environ ayarları)
    # .env dosyasındakilerden önce gelir — böylece admin panelinde seçilen
    # model/temperature geçerli olur.
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=True)
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Put it in a .env file or export it in the environment."
        )

    model = (os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant").strip()
    temperature_raw = (os.getenv("GROQ_TEMPERATURE") or "0.2").strip()
    try:
        temperature = float(temperature_raw)
    except ValueError:
        temperature = 0.2

    return GroqConfig(api_key=api_key, model=model, temperature=temperature)


def build_chat_model(config: Optional[GroqConfig] = None):
    """
    Returns a LangChain chat model backed by Groq.

    Note: This module does not decide risk. It only generates user-facing explanations
    based on RuleEngine output + retrieved evidence (RAG).
    """

    from langchain_groq import ChatGroq

    cfg = config or load_groq_config()
    return ChatGroq(
        api_key=cfg.api_key,
        model=cfg.model,
        temperature=cfg.temperature,
    )


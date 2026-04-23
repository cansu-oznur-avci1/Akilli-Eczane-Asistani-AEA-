from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings


@dataclass(frozen=True)
class RagChunk:
    text: str
    source_path: str
    page: Optional[int] = None


def _make_embeddings(embedding_model: str):
    return HuggingFaceEmbeddings(
        model_name=embedding_model,
    )


def load_vectorstore(
    persist_dir: Path,
    collection_name: str,
    embedding_model: str,
):
    embeddings = _make_embeddings(embedding_model)
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def retrieve(
    query: str,
    *,
    k: int = 4,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
    filter_metadata: dict | None = None,
) -> List[RagChunk]:
    persist_dir = persist_dir or Path(os.getenv("CHROMA_PERSIST_DIR", "vector_db/chroma"))
    collection_name = collection_name or os.getenv("CHROMA_COLLECTION", "aea_kub_kt")
    embedding_model = embedding_model or os.getenv(
        "AEA_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    try:
        vs = load_vectorstore(persist_dir, collection_name, embedding_model)
        search_kwargs_dict = {"k": k}
        if filter_metadata is not None:
            search_kwargs_dict["filter"] = filter_metadata
        retriever = vs.as_retriever(search_kwargs=search_kwargs_dict)
        # Newer LangChain versions: use `invoke()` instead of `get_relevant_documents()`.
        docs = retriever.invoke(query)
    except Exception as e:
        if os.getenv("AEA_DEBUG_RAG", "").strip().lower() in ("1", "true", "yes"):
            print(f"[RAG] retrieve() failed: {e!r}")
        # If the vector DB hasn't been ingested yet, return no evidence.
        return []

    out: List[RagChunk] = []
    for d in docs:
        md = d.metadata or {}
        out.append(
            RagChunk(
                text=d.page_content,
                source_path=str(md.get("source_path", "")),
                page=md.get("page"),
            )
        )
    return out


TIBBI_UYARI_PATTERNS = [
    r"hekiminiz.*?konuşunuz",
    r"doktorunuz.*?danışınız",
    r"eczacınız.*?danışınız",
    r"sağlık.*?profesyonel",
]

def format_evidence(chunks: List[RagChunk], max_chars: int = 3000) -> str:
    parts: List[str] = []
    total = 0
    for c in chunks:
        header = f"[Kaynak: {c.source_path} - sayfa: {c.page if c.page is not None else '?'}]"
        body = c.text.strip().replace("\n", " ")
        # Tıbbi uyarı cümlelerini filtrele
        sentences = body.split(". ")
        filtered = []
        for s in sentences:
            if not any(re.search(p, s, re.IGNORECASE) for p in TIBBI_UYARI_PATTERNS):
                filtered.append(s)
        body = ". ".join(filtered)
        block = f"{header} {body}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)

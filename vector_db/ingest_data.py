from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings


def _iter_pdfs(input_dir: Path, glob_pattern: str) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"PDF input dir not found: {input_dir}")
    # Default glob is case-insensitive for convenience.
    if glob_pattern.lower() == "*.pdf":
        return sorted(
            [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.casefold() == ".pdf"]
        )
    return sorted([p for p in input_dir.rglob(glob_pattern) if p.is_file()])


def _load_documents(pdf_paths: Iterable[Path]) -> list:
    docs: list = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        loaded = loader.load()
        lower = str(pdf_path).casefold()
        provider = "unknown"
        if "titck" in lower:
            provider = "TITCK"
        elif "dicle" in lower:
            provider = "Dicle University"

        # Heuristic: filenames often end with "KUB" / "KT" and contain a drug name.
        doc_type = "unknown"
        if re.search(r"\b(kub|küb|küt)\b", lower):
            doc_type = "KUB"
        elif re.search(r"\b(kkt|kt)\b", lower):
            doc_type = "KT"

        drug_name_norm = _extract_drug_name_norm(pdf_path)

        for d in loaded:
            d.metadata = dict(d.metadata or {})
            d.metadata["source_path"] = str(pdf_path)
            d.metadata["file_name"] = pdf_path.name
            d.metadata["provider"] = provider
            d.metadata["doc_type"] = doc_type
            d.metadata["drug_name_norm"] = drug_name_norm
        docs.extend(loaded)
    return docs


def _extract_drug_name_norm(pdf_path: Path) -> str:
    """
    Best-effort drug-name extraction from file name.
    Used only for metadata; retrieval does not rely on it yet.
    """
    stem = pdf_path.stem.casefold()
    # Drop leading numeric identifiers (e.g. "24584-...").
    stem = re.sub(r"^\s*\d+\s*[-_]\s*", "", stem)
    tokens = re.split(r"[-_ ]+", stem)

    drug_tokens: list[str] = []
    stop_tokens = {"kub", "küb", "küt", "kt", "kkt"}
    unit_stop = {"iu", "i", "u"}  # "i u" in IU-like sequences

    for t in tokens:
        if not t:
            continue
        if t in stop_tokens:
            break
        if t in unit_stop:
            break
        if t.isdigit():
            continue
        # Skip very short tokens that are likely formatting/units.
        if len(t) <= 1:
            continue
        drug_tokens.append(t)
        # Keep only a couple of tokens to avoid swallowing suffixes.
        if len(drug_tokens) >= 3:
            break

    return " ".join(drug_tokens).strip()


def _make_ids(chunks) -> list[str]:
    ids: list[str] = []
    for c in chunks:
        source = str(c.metadata.get("source_path", ""))
        page = str(c.metadata.get("page", ""))
        h = hashlib.sha1(c.page_content.encode("utf-8")).hexdigest()[:12]
        ids.append(f"{h}:{page}:{os.path.basename(source)}")
    return ids


def ingest(
    input_dir: Path,
    persist_dir: Path,
    collection_name: str,
    embedding_model: str,
    glob_pattern: str,
    chunk_size: int,
    chunk_overlap: int,
    clear: bool,
) -> None:
    pdfs = _iter_pdfs(input_dir, glob_pattern)
    if not pdfs:
        print(
            f"No PDFs found in '{input_dir}' (pattern: '{glob_pattern}'). "
            "Please place your KÜB/KT PDFs into the 'pdfs/' folder (or pass --input-dir)."
        )
        return

    raw_docs = _load_documents(pdfs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(raw_docs)
    ids = _make_ids(chunks)

    if clear and persist_dir.exists():
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    vectorstore.add_documents(chunks, ids=ids)

    print(
        f"Ingest done. PDFs={len(pdfs)}, pages={len(raw_docs)}, chunks={len(chunks)}, persist_dir={persist_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AEA PDF -> ChromaDB ingest script")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.getenv("AEA_PDF_INPUT_DIR", "pdfs"),
        help="Folder containing KÜB/KT PDFs",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=os.getenv("CHROMA_PERSIST_DIR", "vector_db/chroma"),
        help="Chroma persistence directory",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=os.getenv("CHROMA_COLLECTION", "aea_kub_kt"),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=os.getenv("AEA_EMBEDDING_MODEL","sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    )
    parser.add_argument("--glob", type=str, default=os.getenv("AEA_PDF_GLOB", "*.pdf"))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("AEA_CHUNK_SIZE", "900")))
    parser.add_argument(
        "--chunk-overlap", type=int, default=int(os.getenv("AEA_CHUNK_OVERLAP", "150"))
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="If set, deletes persist-dir before ingesting (use carefully).",
    )

    args = parser.parse_args()
    ingest(
        input_dir=Path(args.input_dir),
        persist_dir=Path(args.persist_dir),
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        glob_pattern=args.glob,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()


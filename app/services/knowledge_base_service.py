import logging
import re
from functools import lru_cache
from typing import Any

import google.generativeai as genai

from app.core.config import (
    KB_EMBEDDING_MODEL,
    KB_RAW_DOCS_DIR,
    KB_TOP_K,
    NON_US_MARKET_KEYWORDS,
    QDRANT_COLLECTION_NAME,
    US_STOCK_MARKET_KEYWORDS,
)
from app.services.state import KB_BACKEND, KB_COLLECTION, QDRANT_CLIENT
from app.services.text_service import build_kb_search_queries, contains_keyword_variation, normalize_topic_text


logger = logging.getLogger("economy-assistant-bot")


def is_non_us_market_question(user_text: str) -> bool:
    normalized = normalize_topic_text(user_text)
    return any(contains_keyword_variation(normalized, keyword) for keyword in NON_US_MARKET_KEYWORDS)


def is_us_stock_market_question(user_text: str) -> bool:
    normalized = normalize_topic_text(user_text)
    if any(contains_keyword_variation(normalized, keyword) for keyword in US_STOCK_MARKET_KEYWORDS):
        return True
    broad_signals = [("abd", "borsa"), ("amerika", "borsa"), ("abd", "hisse"), ("amerika", "hisse")]
    return any(all(signal in normalized for signal in pair) for pair in broad_signals)


def embed_kb_text(text: str, *, task_type: str) -> list[float]:
    response: dict[str, Any] = genai.embed_content(
        model=KB_EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    embedding = response.get("embedding") or []
    return [float(value) for value in embedding]


def _chunk_raw_text(text: str, chunk_size: int = 1200) -> list[str]:
    clean_text = text.replace("\r", "\n").strip()
    if not clean_text:
        return []

    paragraphs = [paragraph.strip() for paragraph in clean_text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = paragraph
            continue
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", paragraph) if sentence.strip()]
        current = ""
        for sentence in sentences:
            sentence_candidate = f"{current} {sentence}".strip() if current else sentence
            if len(sentence_candidate) <= chunk_size:
                current = sentence_candidate
                continue
            if current:
                chunks.append(current)
            current = sentence[:chunk_size].strip()
        if current:
            chunks.append(current)
        current = ""

    if current:
        chunks.append(current)
    return chunks


@lru_cache(maxsize=1)
def _load_raw_kb_chunks() -> list[tuple[str, str]]:
    if not KB_RAW_DOCS_DIR.exists():
        logger.warning("Raw knowledgebase klasoru bulunamadi: %s", KB_RAW_DOCS_DIR)
        return []

    chunks: list[tuple[str, str]] = []
    for path in sorted(KB_RAW_DOCS_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception:
            logger.warning("Raw knowledgebase dosyasi okunamadi: %s", path, exc_info=True)
            continue
        chunks.extend((path.stem, chunk) for chunk in _chunk_raw_text(text))
    return chunks


def _tokenize_text(text: str) -> set[str]:
    normalized = normalize_topic_text(text)
    stop_words = {
        "acaba",
        "ama",
        "bir",
        "bu",
        "da",
        "de",
        "diye",
        "gibi",
        "hangi",
        "icin",
        "ile",
        "mi",
        "mu",
        "ne",
        "neden",
        "nedir",
        "nasil",
        "olan",
        "olarak",
        "ve",
        "veya",
    }
    return {token for token in re.findall(r"[a-z0-9]+", normalized) if len(token) > 1 and token not in stop_words}


def _search_raw_knowledge_base(query: str) -> list[str]:
    query_variants = build_kb_search_queries(query)
    query_tokens = set().union(*(_tokenize_text(item) for item in query_variants if item.strip()))
    if not query_tokens:
        return []

    scored_chunks: list[tuple[int, int, str]] = []
    seen_chunks: set[str] = set()
    normalized_variants = [normalize_topic_text(item) for item in query_variants if item.strip()]
    for source_name, chunk in _load_raw_kb_chunks():
        if not chunk or chunk in seen_chunks:
            continue
        seen_chunks.add(chunk)
        normalized_chunk = normalize_topic_text(chunk)
        normalized_source_name = normalize_topic_text(source_name)
        chunk_tokens = _tokenize_text(chunk)
        overlap = len(query_tokens & chunk_tokens)
        if overlap == 0:
            continue
        exact_bonus = sum(1 for variant in normalized_variants if variant in normalized_chunk)
        occurrence_bonus = sum(normalized_chunk.count(token) for token in query_tokens)
        filename_bonus = sum(1 for token in query_tokens if token in normalized_source_name)
        leading_bonus = sum(1 for token in query_tokens if token in normalized_chunk[:220])
        score = overlap * 10 + exact_bonus * 20 + occurrence_bonus * 4 + filename_bonus * 15 + leading_bonus * 3
        scored_chunks.append((score, -len(chunk), chunk))

    scored_chunks.sort(reverse=True)
    return [chunk for _, _, chunk in scored_chunks[:KB_TOP_K]]


def _search_qdrant_knowledge_base(query: str) -> list[str]:
    filtered_docs: list[str] = []
    seen_docs: set[str] = set()
    for search_query in build_kb_search_queries(query):
        query_vector = embed_kb_text(search_query, task_type="retrieval_query")
        results = QDRANT_CLIENT.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=KB_TOP_K,
            with_payload=True,
        )
        for point in results:
            payload = point.payload or {}
            document = payload.get("document")
            if not isinstance(document, str) or not document or document in seen_docs:
                continue
            filtered_docs.append(document)
            seen_docs.add(document)
    return filtered_docs


def _search_chroma_knowledge_base(query: str) -> list[str]:
    filtered_docs: list[str] = []
    seen_docs: set[str] = set()
    for search_query in build_kb_search_queries(query):
        result = KB_COLLECTION.query(
            query_texts=[search_query],
            n_results=KB_TOP_K,
            include=["documents", "distances", "metadatas"],
        )
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for document, distance in zip(documents, distances):
            if not document or document in seen_docs:
                continue
            if distance is None or distance <= 2.2:
                filtered_docs.append(document)
                seen_docs.add(document)
    return filtered_docs


def search_knowledge_base(query: str) -> list[str]:
    try:
        if KB_BACKEND == "qdrant" and QDRANT_CLIENT is not None:
            results = _search_qdrant_knowledge_base(query)
            if results:
                return results
        elif KB_COLLECTION is not None:
            results = _search_chroma_knowledge_base(query)
            if results:
                return results
        else:
            logger.warning("Knowledgebase collection yok; raw dokuman fallback kullanilacak.")
    except Exception:
        logger.exception("Vector knowledgebase aramasi basarisiz oldu; raw dokuman fallback kullanilacak.")

    return _search_raw_knowledge_base(query)

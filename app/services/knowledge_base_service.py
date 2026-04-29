import logging
from typing import Any

import google.generativeai as genai

from app.core.config import (
    KB_EMBEDDING_MODEL,
    KB_TOP_K,
    NON_US_MARKET_KEYWORDS,
    QDRANT_COLLECTION_NAME,
    US_STOCK_MARKET_KEYWORDS,
)
from app.core.perf import log_timing
from app.services.state import QDRANT_CLIENT
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


def should_search_knowledge_base(user_text: str) -> bool:
    if is_non_us_market_question(user_text):
        return False
    return is_us_stock_market_question(user_text)


@log_timing()
def embed_kb_text(text: str, *, task_type: str) -> list[float]:
    response: dict[str, Any] = genai.embed_content(
        model=KB_EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    embedding = response.get("embedding") or []
    return [float(value) for value in embedding]


@log_timing()
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


@log_timing()
def search_knowledge_base(query: str) -> list[str]:
    if not should_search_knowledge_base(query):
        logger.info("Soru knowledgebase kapsaminda degil, arama atlandi: %s", query)
        return []

    if QDRANT_CLIENT is None:
        logger.warning("Qdrant istemcisi hazir degil; knowledgebase aramasi yapilamiyor.")
        return []

    try:
        return _search_qdrant_knowledge_base(query)
    except Exception:
        logger.exception("Qdrant knowledgebase aramasi basarisiz oldu.")
        return []

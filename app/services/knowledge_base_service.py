import logging
from typing import Any

import google.generativeai as genai

from app.core.config import KB_EMBEDDING_MODEL, KB_TOP_K, NON_US_MARKET_KEYWORDS, QDRANT_COLLECTION_NAME, US_STOCK_MARKET_KEYWORDS
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


def search_knowledge_base(query: str) -> list[str]:
    if KB_BACKEND == "qdrant" and QDRANT_CLIENT is not None:
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

    if KB_COLLECTION is None:
        logger.warning("Knowledgebase collection yok.")
        return []

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

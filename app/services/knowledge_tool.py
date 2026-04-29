import logging

from app.core.config import UNKNOWN_MESSAGE
from app.core.perf import log_timing
from app.services.ai_service import (
    build_extractive_kb_fallback,
    generate_kb_based_reply,
    generate_kb_context_summary,
)
from app.services.knowledge_base_service import search_knowledge_base, should_search_knowledge_base


logger = logging.getLogger("economy-assistant-bot")


@log_timing()
def answer_with_knowledge_base_tool(chat_id: int, user_text: str, *, summary_only: bool = False) -> str | None:
    if not should_search_knowledge_base(user_text):
        logger.info("Knowledgebase tool atlandi; soru kapsam disi: %s", user_text)
        return None

    context_chunks = search_knowledge_base(user_text)
    if not context_chunks:
        logger.info("Knowledgebase tool sonuc bulamadi: %s", user_text)
        return None

    if summary_only:
        answer = generate_kb_context_summary(chat_id, user_text, context_chunks)
    else:
        answer = generate_kb_based_reply(chat_id, user_text, context_chunks)

    if answer and answer != UNKNOWN_MESSAGE:
        return answer

    if summary_only:
        return None

    fallback_answer = build_extractive_kb_fallback(user_text, context_chunks)
    if fallback_answer and fallback_answer != UNKNOWN_MESSAGE:
        return fallback_answer
    return None

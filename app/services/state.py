import logging
import threading
from typing import Any

import httpx
from google import genai
from google.genai import types

from app.core.config import (
    GEMINI_MODEL_NAME,
    GOOGLE_API_KEY,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    SYSTEM_INSTRUCTION,
)

try:
    from qdrant_client import QdrantClient
except ImportError:  # pragma: no cover
    QdrantClient = None


logger = logging.getLogger("economy-assistant-bot")

GEMINI_CLIENT = genai.Client(api_key=GOOGLE_API_KEY)


class GeminiModel:
    def __init__(self, *, client: genai.Client, model_name: str, system_instruction: str | None = None) -> None:
        self.client = client
        self.model_name = model_name
        self.system_instruction = system_instruction

    def generate_content(self, contents: Any, generation_config: dict[str, Any] | None = None) -> Any:
        config_data = dict(generation_config or {})
        if self.system_instruction:
            config_data.setdefault("system_instruction", self.system_instruction)
        config_data.setdefault("thinking_config", types.ThinkingConfig(thinking_budget=0))
        config = types.GenerateContentConfig(**config_data) if config_data else None
        return self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )


MODEL = GeminiModel(
    client=GEMINI_CLIENT,
    model_name=GEMINI_MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION,
)
HTTP_CLIENT = httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0))

CHAT_MEMORY: dict[int, dict[str, object]] = {}
LAST_ALPHA_VANTAGE_REQUEST_AT = 0.0
ALPHA_VANTAGE_LOCK = threading.Lock()
WEBHOOK_INIT_LOCK = threading.Lock()
WEBHOOK_INITIALIZED = False

QDRANT_CLIENT = None
if QDRANT_URL and QdrantClient is not None:
    try:
        QDRANT_CLIENT = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0)
        QDRANT_CLIENT.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        logger.info("Qdrant collection hazir: %s", QDRANT_COLLECTION_NAME)
    except Exception:
        logger.exception("Qdrant collection baslatilamadi.")
else:
    logger.warning("Qdrant ayari eksik veya qdrant-client kurulu degil. Knowledgebase aramasi devre disi kalacak.")

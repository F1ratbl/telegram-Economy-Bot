import logging
import threading
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors, types

from app.core.config import (
    GEMINI_FALLBACK_MODEL_NAMES,
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


def is_quota_error(exc: genai_errors.APIError) -> bool:
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "").upper()
    message = str(getattr(exc, "message", "") or exc).lower()
    return code == 429 or "RESOURCE_EXHAUSTED" in status or "quota" in message


class GeminiModel:
    def __init__(
        self,
        *,
        client: genai.Client,
        model_name: str,
        fallback_model_names: list[str] | None = None,
        system_instruction: str | None = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.model_names = list(dict.fromkeys([model_name, *(fallback_model_names or [])]))
        self.system_instruction = system_instruction

    def generate_content(self, contents: Any, generation_config: dict[str, Any] | None = None) -> Any:
        config_data = dict(generation_config or {})
        if self.system_instruction:
            config_data.setdefault("system_instruction", self.system_instruction)
        last_quota_error: genai_errors.APIError | None = None
        for model_name in self.model_names:
            model_config_data = dict(config_data)
            if "2.5" in model_name:
                model_config_data.setdefault("thinking_config", types.ThinkingConfig(thinking_budget=0))
            try:
                return self._generate_with_config(contents, model_config_data, model_name)
            except genai_errors.APIError as exc:
                if not is_quota_error(exc):
                    raise
                last_quota_error = exc
                logger.warning("Gemini modeli kota hatasi verdi, siradaki model denenecek: %s", model_name)
        if last_quota_error:
            raise last_quota_error
        raise RuntimeError("Gemini modeli tanimli degil.")

    def _generate_with_config(self, contents: Any, config_data: dict[str, Any], model_name: str) -> Any:
        config = types.GenerateContentConfig(**config_data) if config_data else None
        try:
            return self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except genai_errors.APIError as exc:
            message = str(getattr(exc, "message", "") or exc).lower()
            if "thinking" not in message or "thinking_config" not in config_data:
                raise
            logger.warning("Model thinking_config desteklemedi; ayni istek thinking olmadan tekrar deneniyor.")
            retry_config_data = dict(config_data)
            retry_config_data.pop("thinking_config", None)
            retry_config = types.GenerateContentConfig(**retry_config_data) if retry_config_data else None
            return self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=retry_config,
            )


MODEL = GeminiModel(
    client=GEMINI_CLIENT,
    model_name=GEMINI_MODEL_NAME,
    fallback_model_names=GEMINI_FALLBACK_MODEL_NAMES,
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

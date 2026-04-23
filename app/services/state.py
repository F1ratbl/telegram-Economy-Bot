import logging
import threading

import google.generativeai as genai
import httpx

from app.core.config import (
    GEMINI_MODEL_NAME,
    GOOGLE_API_KEY,
    KB_COLLECTION_NAME,
    KB_DB_DIR,
    IS_VERCEL,
    SYSTEM_INSTRUCTION,
)

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None


logger = logging.getLogger("economy-assistant-bot")

genai.configure(api_key=GOOGLE_API_KEY)
MODEL = genai.GenerativeModel(
    model_name=GEMINI_MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION,
)
HTTP_CLIENT = httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0))

CHAT_MEMORY: dict[int, dict[str, object]] = {}
LAST_ALPHA_VANTAGE_REQUEST_AT = 0.0
ALPHA_VANTAGE_LOCK = threading.Lock()
WEBHOOK_INIT_LOCK = threading.Lock()
WEBHOOK_INITIALIZED = False

KB_COLLECTION = None
if chromadb is None:
    logger.warning("chromadb kurulu degil. Knowledgebase aramasi devre disi kalacak.")
else:
    try:
        kb_client = chromadb.PersistentClient(path=str(KB_DB_DIR))
        logger.info("Yerel Chroma PersistentClient kullaniliyor.")
        try:
            KB_COLLECTION = kb_client.get_collection(name=KB_COLLECTION_NAME)
        except Exception:
            if IS_VERCEL:
                raise
            KB_COLLECTION = kb_client.get_or_create_collection(name=KB_COLLECTION_NAME)
        logger.info("Knowledgebase collection hazir: %s", KB_COLLECTION_NAME)
    except Exception:
        logger.exception("Knowledgebase collection baslatilamadi.")

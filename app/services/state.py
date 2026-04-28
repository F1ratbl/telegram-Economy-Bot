import logging
import threading

import google.generativeai as genai
import httpx

from app.core.config import (
    GEMINI_MODEL_NAME,
    GOOGLE_API_KEY,
    KB_COLLECTION_NAME,
    KB_DB_DIR,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    IS_VERCEL,
    SYSTEM_INSTRUCTION,
)

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None

try:
    from qdrant_client import QdrantClient
except ImportError:  # pragma: no cover
    QdrantClient = None


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
KB_BACKEND = None
QDRANT_CLIENT = None
if QDRANT_URL and QdrantClient is not None:
    try:
        QDRANT_CLIENT = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0)
        QDRANT_CLIENT.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        KB_BACKEND = "qdrant"
        logger.info("Qdrant collection hazir: %s", QDRANT_COLLECTION_NAME)
    except Exception:
        logger.exception("Qdrant collection baslatilamadi.")

if KB_BACKEND is None and chromadb is None:
    logger.warning("Ne qdrant ne de chromadb hazir. Knowledgebase aramasi devre disi kalacak.")
elif KB_BACKEND is None:
    try:
        kb_client = chromadb.PersistentClient(path=str(KB_DB_DIR))
        logger.info("Yerel Chroma PersistentClient kullaniliyor.")
        try:
            KB_COLLECTION = kb_client.get_collection(name=KB_COLLECTION_NAME)
        except Exception:
            if IS_VERCEL:
                raise
            KB_COLLECTION = kb_client.get_or_create_collection(name=KB_COLLECTION_NAME)
        KB_BACKEND = "chroma"
        logger.info("Knowledgebase collection hazir: %s", KB_COLLECTION_NAME)
    except Exception:
        logger.exception("Knowledgebase collection baslatilamadi.")

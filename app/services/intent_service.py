import json
import logging
import re

from app.core.perf import log_timing
from app.services.memory_service import format_memory_context
from app.services.state import MODEL
from app.services.text_service import sanitize_reply_text


logger = logging.getLogger("economy-assistant-bot")

SUPPORTED_INTENTS = {
    "greeting",
    "how_are_you",
    "acknowledgement",
    "capability_question",
    "ask_stored_name",
    "name_addressing_request",
    "general_economy",
    "unknown",
}


def _extract_json_object(text: str) -> dict[str, str] | None:
    cleaned = sanitize_reply_text(text)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return {str(key): str(value) for key, value in payload.items()}


@log_timing()
def classify_user_intent(chat_id: int, user_text: str) -> str:
    memory_context = format_memory_context(chat_id)
    prompt = f"""
Kullanicinin mesajini asagidaki intentlerden sadece birine siniflandir:
- greeting
- how_are_you
- acknowledgement
- capability_question
- ask_stored_name
- name_addressing_request
- general_economy
- unknown

Kurallar:
- Sadece JSON don.
- JSON formati tam olarak soyle olsun: {{"intent":"<intent>"}}
- Aciklama yazma.
- Eger kullanici selam veriyorsa greeting sec.
- Eger kullanici nasil oldugunu soruyorsa how_are_you sec.
- Eger kullanici tamam, anladim, peki, olur, eyvallah, tesekkur, aferin, helal, cok iyisin gibi kisa bir onay, ovgu veya konusma devam mesaji veriyorsa acknowledgement sec.
- Eger botun hangi konularda yardimci olabilecegini soruyorsa capability_question sec.
- Eger kayitli adini soruyorsa ask_stored_name sec.
- Eger bundan sonra adiyla, ismiyle veya benzer sekilde hitap edilmesini istiyorsa name_addressing_request sec.
- Eger mesaj genel ekonomi, piyasa, yatirim veya finans aciklamasi istiyorsa general_economy sec.
- Canli fiyat, kur veya tool gerektiren sorularda unknown sec.
- Kullanici adini yeni veriyorsa unknown sec.

Bellek:
{memory_context}

Mesaj:
{user_text}
""".strip()
    try:
        response = MODEL.generate_content(prompt, generation_config={"max_output_tokens": 80})
        response_text = getattr(response, "text", None) or ""
        parsed = _extract_json_object(response_text)
        intent = (parsed or {}).get("intent", "unknown").strip()
        if intent in SUPPORTED_INTENTS:
            return intent
    except Exception:
        logger.exception("Gemini intent siniflandirmasi basarisiz oldu.")
    return "unknown"

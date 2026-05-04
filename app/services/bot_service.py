import logging
from pathlib import Path

from app.core.config import UNKNOWN_MESSAGE, WEBHOOK_BASE_URL
from app.core.perf import log_timing, timed_block
from app.services.ai_service import delete_uploaded_gemini_file, generate_general_reply, transcribe_voice_to_text
from app.services.intent_service import classify_user_intent
from app.services.knowledge_base_service import should_search_knowledge_base
from app.services.knowledge_tool import answer_with_knowledge_base_tool
from app.services.market_service import answer_with_market_tool
from app.services.memory_service import detect_user_name, get_chat_memory, remember_exchange
from app.services.state import WEBHOOK_INIT_LOCK
import app.services.state as state
from app.services.telegram_service import (
    download_telegram_voice,
    send_text_and_voice_reply,
    send_text_message,
    telegram_api_request,
)
from app.services.text_service import (
    compact_direct_market_reply,
    is_asking_stored_name,
    is_capability_question,
    is_general_economy_question,
    is_greeting_question,
    is_how_are_you_question,
    is_name_addressing_request,
    is_smalltalk_question,
    strip_market_source_details,
)


logger = logging.getLogger("economy-assistant-bot")


def is_start_command(message: dict[str, object]) -> bool:
    text = (message.get("text") or "").strip()
    if not text:
        return False

    entities = message.get("entities") or []
    if not isinstance(entities, list) or not entities:
        return False

    first_entity = entities[0]
    if not isinstance(first_entity, dict):
        return False
    if first_entity.get("type") != "bot_command" or first_entity.get("offset") != 0:
        return False

    command_length = first_entity.get("length")
    if not isinstance(command_length, int) or command_length <= 0:
        return False

    command_text = text[:command_length].split("@", 1)[0].lower()
    return command_text == "/start"


def initialize_webhook() -> None:
    if state.WEBHOOK_INITIALIZED or not WEBHOOK_BASE_URL:
        return
    with WEBHOOK_INIT_LOCK:
        if state.WEBHOOK_INITIALIZED or not WEBHOOK_BASE_URL:
            return
        webhook_url = f"{WEBHOOK_BASE_URL.rstrip('/')}/webhook"
        try:
            telegram_api_request("setWebhook", data={"url": webhook_url})
            logger.info("Webhook ayarlandi: %s", webhook_url)
            state.WEBHOOK_INITIALIZED = True
        except Exception:
            logger.exception("Webhook otomatik olarak ayarlanamadi.")


def safe_delete_file(file_path: Path | None) -> None:
    if not file_path:
        return
    try:
        file_path.unlink(missing_ok=True)
    except Exception:
        logger.warning("Gecici dosya silinemedi: %s", file_path, exc_info=True)


def combine_tool_and_kb_answers(tool_answer: str, kb_answer: str) -> str:
    cleaned_tool = tool_answer.strip().rstrip(".")
    cleaned_kb = kb_answer.strip()
    if cleaned_kb.endswith("."):
        return f"{cleaned_tool}. {cleaned_kb}"
    return f"{cleaned_tool}. {cleaned_kb}."


def build_capability_reply(chat_id: int) -> str:
    user_name = get_chat_memory(chat_id).get("name")
    prefix = f"{user_name}, " if user_name else ""
    return (
        f"{prefix}ben daha cok ABD borsasi islemleriyle ilgili konularda yardimci oluyorum. "
        "NYSE ve Nasdaq farki, seans saatleri, emir tipleri, cash account ve margin account farki, "
        "short selling, pattern day trader kurali gibi basliklarda bilgi verebilirim. "
        "Ayrica dolar, euro, petrol ve bazi ABD endeksleri icin canli veri de paylasabiliyorum."
    )


def build_name_ack_reply(chat_id: int) -> str:
    user_name = get_chat_memory(chat_id).get("name")
    if not user_name:
        return "Ismini kaydedemedim ama istersen tekrar yaz, bir sonraki mesajlarda onunla hitap edeyim."
    return f"Memnun oldum {user_name}. Bundan sonra uygun oldugunda sana adinla hitap ederim."


def build_current_name_reply(chat_id: int) -> str:
    user_name = get_chat_memory(chat_id).get("name")
    if not user_name:
        return "Henuz adini kaydetmedim. Istersen benim adim Firat diye yaz, hafizaya alayim."
    return f"Adin {user_name}."


def build_name_addressing_reply(chat_id: int) -> str:
    user_name = get_chat_memory(chat_id).get("name")
    if not user_name:
        return "Memnuniyetle, ama once adini bilmem gerekiyor. Ornek olarak benim adim Firat yazabilirsin."
    return f"Tabii {user_name}. Bundan sonra uygun oldugunda sana adinla hitap ederim."


def build_greeting_reply(chat_id: int) -> str:
    user_name = get_chat_memory(chat_id).get("name")
    prefix = f"{user_name}, " if user_name else ""
    return f"{prefix}merhaba, hos geldin. Size nasıl yardımcı olabilirim."


def build_how_are_you_reply(chat_id: int) -> str:
    user_name = get_chat_memory(chat_id).get("name")
    prefix = f"{user_name}, " if user_name else ""
    return (
        f"{prefix}iyiyim, tesekkur ederim. "
        "Ekonomi, doviz, petrol ve ABD borsasi konularinda yardimci olabilirim. "
        "Size nasıl yardımcı olabilirim."
    )


def build_gemini_intent_reply(chat_id: int, user_text: str) -> str | None:
    intent = classify_user_intent(chat_id, user_text)
    if intent == "capability_question":
        return build_capability_reply(chat_id)
    if intent == "ask_stored_name":
        return build_current_name_reply(chat_id)
    if intent == "name_addressing_request":
        return build_name_addressing_reply(chat_id)
    if intent == "greeting":
        return build_greeting_reply(chat_id)
    if intent == "how_are_you":
        return build_how_are_you_reply(chat_id)
    if intent == "general_economy" and not should_search_knowledge_base(user_text):
        return generate_general_reply(chat_id, user_text)
    return None


@log_timing()
def answer_question_with_kb(chat_id: int, user_text: str) -> str:
    normalized_user_text = user_text.strip()
    if is_capability_question(normalized_user_text):
        return build_capability_reply(chat_id)

    if is_asking_stored_name(normalized_user_text):
        return build_current_name_reply(chat_id)

    if is_name_addressing_request(normalized_user_text):
        return build_name_addressing_reply(chat_id)

    if is_greeting_question(normalized_user_text):
        return build_greeting_reply(chat_id)

    if is_how_are_you_question(normalized_user_text):
        return build_how_are_you_reply(chat_id)

    if is_smalltalk_question(normalized_user_text):
        return build_how_are_you_reply(chat_id)

    if detect_user_name(normalized_user_text) and len(normalized_user_text.split()) <= 6:
        return build_name_ack_reply(chat_id)

    tool_answer = answer_with_market_tool(user_text)
    if tool_answer is not None:
        if should_search_knowledge_base(user_text):
            kb_answer = answer_with_knowledge_base_tool(chat_id, user_text, summary_only=True)
            if kb_answer:
                return combine_tool_and_kb_answers(tool_answer, kb_answer)
        return tool_answer

    gemini_intent_reply = build_gemini_intent_reply(chat_id, normalized_user_text)
    if gemini_intent_reply is not None:
        return gemini_intent_reply

    if not should_search_knowledge_base(user_text):
        if is_general_economy_question(user_text):
            return generate_general_reply(chat_id, user_text)
        return UNKNOWN_MESSAGE

    kb_answer = answer_with_knowledge_base_tool(chat_id, user_text)
    if kb_answer:
        return kb_answer

    return UNKNOWN_MESSAGE


@log_timing()
def process_update(update: dict[str, object]) -> None:
    input_audio_path: Path | None = None
    output_audio_path: Path | None = None
    gemini_file_name: str | None = None
    chat_id: int | None = None

    try:
        message = update.get("message") or update.get("edited_message")
        if not message:
            logger.info("Mesaj icermeyen update atlandi: %s", update)
            return

        if is_start_command(message):
            logger.info("/start komutu alindi, cevap gonderilmeden atlandi.")
            return

        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if not chat_id:
            logger.warning("chat_id bulunamadi: %s", update)
            return

        user_text = (message.get("text") or "").strip()
        voice_payload = message.get("voice")

        detected_name = detect_user_name(user_text) if user_text else None
        if detected_name:
            get_chat_memory(chat_id)["name"] = detected_name
            logger.info("Kullanici adi hafizaya kaydedildi: %s", detected_name)

        if user_text:
            with timed_block("process_update.answer_text"):
                reply_text = answer_question_with_kb(chat_id, user_text)
        elif voice_payload:
            with timed_block("process_update.voice_download"):
                input_audio_path = download_telegram_voice(voice_payload["file_id"])
            with timed_block("process_update.voice_transcription"):
                transcribed_text, gemini_file_name = transcribe_voice_to_text(input_audio_path)
            detected_name = detect_user_name(transcribed_text)
            if detected_name:
                get_chat_memory(chat_id)["name"] = detected_name
                logger.info("Kullanici adi sesli mesajdan hafizaya kaydedildi: %s", detected_name)
            with timed_block("process_update.answer_transcribed_text"):
                reply_text = answer_question_with_kb(chat_id, transcribed_text)
            user_text = transcribed_text
        else:
            send_text_message(
                chat_id,
                "Lutfen bana yazi veya sesli mesaj gonder. Sana hem yazi hem ses olarak donus yapayim.",
            )
            return

        reply_text = strip_market_source_details(reply_text)
        reply_text = compact_direct_market_reply(user_text or "", reply_text)
        reply_text = reply_text.strip() or "Uzgunum, bu mesaja su anda anlamli bir yanit uretemedim."

        if user_text:
            remember_exchange(chat_id, user_text, reply_text)
        elif voice_payload:
            remember_exchange(chat_id, "[Sesli mesaj]", reply_text)

        with timed_block("process_update.reply_delivery"):
            output_audio_path = send_text_and_voice_reply(chat_id, reply_text)
    except Exception:
        logger.exception("Update islenirken hata olustu.")
        if chat_id:
            try:
                send_text_message(
                    chat_id,
                    "Uzgunum, istegini islerken bir hata olustu. Birazdan tekrar deneyebilirsin.",
                )
            except Exception:
                logger.exception("Hata mesaji kullaniciya gonderilemedi.")
    finally:
        safe_delete_file(input_audio_path)
        safe_delete_file(output_audio_path)
        delete_uploaded_gemini_file(gemini_file_name)
def start_background_update(update: dict[str, object]) -> None:
    process_update(update)

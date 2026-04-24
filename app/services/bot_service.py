import logging
from pathlib import Path

from app.core.config import UNKNOWN_MESSAGE, WEBHOOK_BASE_URL
from app.services.ai_service import (
    delete_uploaded_gemini_file,
    generate_kb_based_reply,
    generate_kb_context_summary,
    transcribe_voice_to_text,
)
from app.services.knowledge_base_service import (
    is_non_us_market_question,
    is_us_stock_market_question,
    search_knowledge_base,
)
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
from app.services.text_service import is_capability_question


logger = logging.getLogger("economy-assistant-bot")


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
    return f"{tool_answer} {kb_answer}".strip()


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


def answer_question_with_kb(chat_id: int, user_text: str) -> str:
    normalized_user_text = user_text.strip()
    if is_capability_question(normalized_user_text):
        return build_capability_reply(chat_id)

    if detect_user_name(normalized_user_text) and len(normalized_user_text.split()) <= 6:
        return build_name_ack_reply(chat_id)

    tool_answer = answer_with_market_tool(user_text)
    if tool_answer is not None:
        if is_us_stock_market_question(user_text):
            context_chunks = search_knowledge_base(user_text)
            if context_chunks:
                kb_answer = generate_kb_context_summary(chat_id, user_text, context_chunks)
                if kb_answer and kb_answer != UNKNOWN_MESSAGE:
                    return combine_tool_and_kb_answers(tool_answer, kb_answer)
        return tool_answer

    if is_non_us_market_question(user_text):
        return UNKNOWN_MESSAGE
    if not is_us_stock_market_question(user_text):
        return UNKNOWN_MESSAGE
    context_chunks = search_knowledge_base(user_text)
    if not context_chunks:
        return UNKNOWN_MESSAGE
    answer = generate_kb_based_reply(chat_id, user_text, context_chunks)
    if not answer:
        return UNKNOWN_MESSAGE
    return answer


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
            reply_text = answer_question_with_kb(chat_id, user_text)
        elif voice_payload:
            input_audio_path = download_telegram_voice(voice_payload["file_id"])
            transcribed_text, gemini_file_name = transcribe_voice_to_text(input_audio_path)
            detected_name = detect_user_name(transcribed_text)
            if detected_name:
                get_chat_memory(chat_id)["name"] = detected_name
                logger.info("Kullanici adi sesli mesajdan hafizaya kaydedildi: %s", detected_name)
            reply_text = answer_question_with_kb(chat_id, transcribed_text)
            user_text = transcribed_text
        else:
            send_text_message(
                chat_id,
                "Lutfen bana yazi veya sesli mesaj gonder. Sana hem yazi hem ses olarak donus yapayim.",
            )
            return

        reply_text = reply_text.strip() or "Uzgunum, bu mesaja su anda anlamli bir yanit uretemedim."

        if user_text:
            remember_exchange(chat_id, user_text, reply_text)
        elif voice_payload:
            remember_exchange(chat_id, "[Sesli mesaj]", reply_text)

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

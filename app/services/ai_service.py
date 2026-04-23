import time
from pathlib import Path
from typing import Any

import google.generativeai as genai

from app.core.config import MAX_OUTPUT_TOKENS, UNKNOWN_MESSAGE
from app.services.memory_service import format_memory_context, get_chat_memory
from app.services.state import MODEL
from app.services.text_service import sanitize_reply_text


def wait_for_uploaded_file(file_name: str, timeout_seconds: int = 120):
    deadline = time.monotonic() + timeout_seconds
    while True:
        uploaded_file = genai.get_file(file_name)
        state = getattr(getattr(uploaded_file, "state", None), "name", "ACTIVE")
        if state == "ACTIVE":
            return uploaded_file
        if state == "FAILED":
            raise RuntimeError("Gemini ses dosyasini isleyemedi.")
        if time.monotonic() > deadline:
            raise TimeoutError("Gemini ses dosyasi islenirken zaman asimina ugradi.")
        time.sleep(2)


def extract_response_text(response: Any) -> str:
    text = (getattr(response, "text", None) or "").strip()
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    chunks: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(part_text)
    final_text = "\n".join(chunks).strip()
    if final_text:
        return final_text
    raise RuntimeError("Gemini metin yaniti uretemedi.")


def generate_kb_based_reply(chat_id: int, user_text: str, context_chunks: list[str]) -> str:
    context_text = "\n\n".join(f"Belge Parcasi {index + 1}:\n{chunk}" for index, chunk in enumerate(context_chunks))
    user_name = get_chat_memory(chat_id).get("name")
    hitap = f"Kullanicinin adi: {user_name}" if user_name else "Kullanicinin adi bilinmiyor."
    prompt = f"""
Asagida ABD Borsasi Islemleri bilgi tabanindan getirilen belge parcalari bulunuyor.
Sadece bu belge parcalarina dayanarak cevap ver.
Belgede olmayan hicbir bilgiyi ekleme.
Eger belgeler soruyu cevaplamak icin yetersizse tam olarak su cumleyi ver: {UNKNOWN_MESSAGE}
Yaniti Turkce ver.
Yildiz kullanma.
Markdown kullanma.
{hitap}

Bilgi tabani:
{context_text}

Soru:
{user_text}
""".strip()
    response = MODEL.generate_content(prompt, generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
    return sanitize_reply_text(extract_response_text(response))


def generate_kb_context_summary(chat_id: int, user_text: str, context_chunks: list[str]) -> str:
    context_text = "\n\n".join(f"Belge Parcasi {index + 1}:\n{chunk}" for index, chunk in enumerate(context_chunks))
    user_name = get_chat_memory(chat_id).get("name")
    hitap = f"Kullanicinin adi: {user_name}" if user_name else "Kullanicinin adi bilinmiyor."
    prompt = f"""
Asagida ABD Borsasi Islemleri bilgi tabanindan getirilen belge parcalari bulunuyor.
Kullanicinin sorusuyla ilgili olarak bu belgelerden cikan arka plan bilgisini kisaca ozetle.
Sadece belgeye dayali bilgi ver.
Belgede olmayan guncel fiyat, hedef veya yorum ekleme.
Eger belgeler hic ilgili degilse tam olarak su cumleyi ver: {UNKNOWN_MESSAGE}
Yaniti Turkce ver.
Yildiz kullanma.
Markdown kullanma.
En fazla 3 cumle kur.
{hitap}

Bilgi tabani:
{context_text}

Kullanicinin sorusu:
{user_text}
""".strip()
    response = MODEL.generate_content(prompt, generation_config={"max_output_tokens": 220})
    return sanitize_reply_text(extract_response_text(response))


def transcribe_voice_to_text(audio_path: Path) -> tuple[str, str]:
    uploaded_file = genai.upload_file(path=audio_path, mime_type="audio/ogg")
    ready_file = wait_for_uploaded_file(uploaded_file.name)
    response = MODEL.generate_content(
        [
            """
Bu sesli mesaji sadece duz metin olarak yaz.
Ek yorum ekleme.
Mumkun oldugunca dogrudan kullanicinin sordugu soruyu cikart.
""".strip(),
            ready_file,
        ],
        generation_config={"max_output_tokens": 200},
    )
    return sanitize_reply_text(extract_response_text(response)), uploaded_file.name


def delete_uploaded_gemini_file(file_name: str | None) -> None:
    if not file_name:
        return
    try:
        genai.delete_file(file_name)
    except Exception:
        import logging

        logging.getLogger("economy-assistant-bot").warning(
            "Gemini yuklenen dosyasi silinemedi: %s", file_name, exc_info=True
        )

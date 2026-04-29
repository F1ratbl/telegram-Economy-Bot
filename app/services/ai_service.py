import time
from pathlib import Path
import re
from typing import Any

import google.generativeai as genai

from app.core.perf import log_timing
from app.core.config import MAX_OUTPUT_TOKENS, UNKNOWN_MESSAGE
from app.services.memory_service import get_chat_memory
from app.services.state import MODEL
from app.services.text_service import sanitize_reply_text
import logging


logger = logging.getLogger("economy-assistant-bot")


@log_timing()
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


def extract_finish_reasons(response: Any) -> list[str]:
    candidates = getattr(response, "candidates", None) or []
    reasons: list[str] = []
    for candidate in candidates:
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is None:
            continue
        reason_name = getattr(finish_reason, "name", None)
        reasons.append(str(reason_name or finish_reason))
    return reasons


def _build_user_name_context(chat_id: int) -> str:
    user_name = get_chat_memory(chat_id).get("name")
    return f"Kullanicinin adi: {user_name}" if user_name else "Kullanicinin adi bilinmiyor."


@log_timing()
def _generate_text(prompt: str, *, max_output_tokens: int) -> str:
    response = MODEL.generate_content(prompt, generation_config={"max_output_tokens": max_output_tokens})
    finish_reasons = extract_finish_reasons(response)
    if finish_reasons:
        logger.info(
            "Gemini finish_reason=%s | max_output_tokens=%s",
            ",".join(finish_reasons),
            max_output_tokens,
        )
        if any(reason.upper() == "MAX_TOKENS" for reason in finish_reasons):
            logger.warning("Gemini cevabi token sinirina carparak bitmis olabilir.")
    return sanitize_reply_text(extract_response_text(response))


def _tokenize_for_overlap(text: str) -> set[str]:
    lowered = text.lower()
    lowered = lowered.translate(str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"}))
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    stop_words = {
        "acaba",
        "ama",
        "bir",
        "bu",
        "da",
        "de",
        "dedir",
        "diye",
        "gibi",
        "hangi",
        "icin",
        "ile",
        "mi",
        "mu",
        "muhtemelen",
        "ne",
        "neden",
        "nedir",
        "nasil",
        "olan",
        "olarak",
        "ve",
        "veya",
    }
    return {token for token in tokens if len(token) > 1 and token not in stop_words}


def _is_source_like_sentence(sentence: str) -> bool:
    normalized = sentence.lower()
    source_markers = [
        "kaynak ozeti",
        "investor.gov",
        "markets overview",
        "trading information",
        "stocks faq",
    ]
    return any(marker in normalized for marker in source_markers)


@log_timing()
def build_extractive_kb_fallback(user_text: str, context_chunks: list[str]) -> str:
    question_tokens = _tokenize_for_overlap(user_text)
    best_sentences: list[tuple[int, int, str]] = []

    for chunk_index, chunk in enumerate(context_chunks):
        sentences = [
            sanitize_reply_text(sentence)
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk)
            if sanitize_reply_text(sentence)
        ]
        for sentence_index, sentence in enumerate(sentences):
            if _is_source_like_sentence(sentence):
                continue
            sentence_tokens = _tokenize_for_overlap(sentence)
            overlap_score = len(question_tokens & sentence_tokens)
            if overlap_score == 0 and question_tokens:
                continue
            best_sentences.append((overlap_score, -(chunk_index * 100 + sentence_index), sentence))

    if not best_sentences:
        first_chunk = next((chunk for chunk in context_chunks if chunk.strip()), "")
        fallback_sentences = [
            sanitize_reply_text(sentence)
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", first_chunk)
            if sanitize_reply_text(sentence) and not _is_source_like_sentence(sanitize_reply_text(sentence))
        ]
        best_text = " ".join(fallback_sentences[:2]).strip()
        return best_text or UNKNOWN_MESSAGE

    best_sentences.sort(reverse=True)
    selected: list[str] = []
    seen_sentences: set[str] = set()
    for _, _, sentence in best_sentences:
        if sentence in seen_sentences:
            continue
        selected.append(sentence)
        seen_sentences.add(sentence)
        if len(selected) == 2:
            break

    return " ".join(selected).strip() or UNKNOWN_MESSAGE


@log_timing()
def generate_kb_based_reply(chat_id: int, user_text: str, context_chunks: list[str]) -> str:
    context_text = "\n\n".join(f"Belge Parcasi {index + 1}:\n{chunk}" for index, chunk in enumerate(context_chunks))
    hitap = _build_user_name_context(chat_id)
    prompt = f"""
Asagida ABD Borsasi Islemleri bilgi tabanindan getirilen belge parcalari bulunuyor.
Sadece bu belge parcalarina dayanarak cevap ver.
Belgede olmayan hicbir bilgiyi ekleme.
Eger belgeler soruyu cevaplamak icin yetersizse tam olarak su cumleyi ver: {UNKNOWN_MESSAGE}
Yaniti Turkce ver.
Yildiz kullanma.
Markdown kullanma.
Dogal, akici ve insan gibi yaz.
Gereksiz baslik kullanma.
Mumkunse dogrudan cevapla basla.
Yarim cumle birakma.
Tek parca, tamamlanmis bir cevap ver.
{hitap}

Bilgi tabani:
{context_text}

Soru:
{user_text}
""".strip()
    return _generate_text(prompt, max_output_tokens=MAX_OUTPUT_TOKENS)


@log_timing()
def generate_kb_context_summary(chat_id: int, user_text: str, context_chunks: list[str]) -> str:
    context_text = "\n\n".join(f"Belge Parcasi {index + 1}:\n{chunk}" for index, chunk in enumerate(context_chunks))
    hitap = _build_user_name_context(chat_id)
    prompt = f"""
Asagida ABD Borsasi Islemleri bilgi tabanindan getirilen belge parcalari bulunuyor.
Kullanicinin sorusundaki aciklayici kismi hedef alarak bu belgelerden cikan arka plan bilgisini kisaca ozetle.
Sadece belgeye dayali bilgi ver.
Belgede olmayan guncel fiyat, hedef veya yorum ekleme.
Eger belgeler hic ilgili degilse tam olarak su cumleyi ver: {UNKNOWN_MESSAGE}
Yaniti Turkce ver.
Yildiz kullanma.
Markdown kullanma.
Dogal, akici ve insan gibi yaz.
Soruda canli fiyat ile aciklama bir aradaysa sadece aciklama kismini ozetle.
En fazla 3 cumle kur.
Yarim cumle birakma.
Tek parca, tamamlanmis bir cevap ver.
{hitap}

Bilgi tabani:
{context_text}

Kullanicinin sorusu:
{user_text}
""".strip()
    return _generate_text(prompt, max_output_tokens=220)


@log_timing()
def verbalize_market_reply(user_text: str, facts: dict[str, str]) -> str:
    facts_text = "\n".join(f"- {key}: {value}" for key, value in facts.items())
    prompt = f"""
Asagidaki canli piyasa verilerini kullanarak kullaniciya dogal, akici ve insan gibi kisa bir Turkce cevap ver.

Kurallar:
- Verilen sayilari, tarihleri ve sembolleri hic degistirme.
- Yeni veri uydurma.
- Tek paragraf yaz.
- Yildiz kullanma.
- Markdown kullanma.
- Yarim cumle birakma.
- Gereksiz aciklama yapma.
- Kullanici dogrudan fiyat sorduysa cevaba dogrudan fiyatla basla.
- En fazla 3 cumle kur.

Kullanicinin sorusu:
{user_text}

Veriler:
{facts_text}
""".strip()
    return _generate_text(prompt, max_output_tokens=180)


@log_timing()
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
        logger.warning(
            "Gemini yuklenen dosyasi silinemedi: %s", file_name, exc_info=True
        )

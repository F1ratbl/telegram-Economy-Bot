import logging
import uuid
from pathlib import Path
from typing import Any

from gtts import gTTS

from app.core.config import TELEGRAM_API_BASE, TELEGRAM_FILE_BASE, TEMP_AUDIO_DIR
from app.services.state import HTTP_CLIENT


logger = logging.getLogger("economy-assistant-bot")


def telegram_api_request(
    method: str,
    *,
    data: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = HTTP_CLIENT.post(
        f"{TELEGRAM_API_BASE}/{method}",
        data=data,
        files=files,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram API hatasi ({method}): {payload}")
    return payload["result"]


def send_text_message(chat_id: int, text: str) -> None:
    telegram_api_request("sendMessage", data={"chat_id": str(chat_id), "text": text})


def send_voice_message(chat_id: int, audio_path: Path) -> None:
    with audio_path.open("rb") as audio_file:
        telegram_api_request(
            "sendVoice",
            data={"chat_id": str(chat_id)},
            files={"voice": (audio_path.name, audio_file, "audio/mpeg")},
        )


def send_audio_message(chat_id: int, audio_path: Path) -> None:
    with audio_path.open("rb") as audio_file:
        telegram_api_request(
            "sendAudio",
            data={"chat_id": str(chat_id)},
            files={"audio": (audio_path.name, audio_file, "audio/mpeg")},
        )


def synthesize_turkish_speech(text: str) -> Path:
    output_path = TEMP_AUDIO_DIR / f"telegram_reply_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang="tr", slow=False)
    tts.save(str(output_path))
    return output_path


def send_text_and_voice_reply(chat_id: int, reply_text: str) -> Path | None:
    send_text_message(chat_id, reply_text)
    output_audio_path = synthesize_turkish_speech(reply_text)
    try:
        send_voice_message(chat_id, output_audio_path)
        logger.info("Sesli yanit sendVoice ile gonderildi.")
    except Exception:
        logger.warning("sendVoice basarisiz oldu, sendAudio denenecek.", exc_info=True)
        send_audio_message(chat_id, output_audio_path)
        logger.info("Sesli yanit sendAudio fallback ile gonderildi.")
    return output_audio_path


def download_telegram_voice(file_id: str) -> Path:
    file_result = telegram_api_request("getFile", data={"file_id": file_id})
    file_path = file_result["file_path"]
    download_url = f"{TELEGRAM_FILE_BASE}/{file_path}"
    local_path = TEMP_AUDIO_DIR / f"telegram_input_{uuid.uuid4().hex}.ogg"
    with HTTP_CLIENT.stream("GET", download_url) as response:
        response.raise_for_status()
        with local_path.open("wb") as output_file:
            for chunk in response.iter_bytes():
                output_file.write(chunk)
    return local_path

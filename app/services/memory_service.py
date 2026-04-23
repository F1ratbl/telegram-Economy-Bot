import re
from typing import Any

from app.core.config import MAX_MEMORY_TURNS
from app.services.state import CHAT_MEMORY


def get_chat_memory(chat_id: int) -> dict[str, Any]:
    memory = CHAT_MEMORY.setdefault(chat_id, {"name": None, "history": []})
    memory.setdefault("history", [])
    memory.setdefault("name", None)
    return memory


def detect_user_name(user_text: str) -> str | None:
    patterns = [
        r"\bad[ıi]m\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)",
        r"\bismim\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)",
        r"\bben\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)",
        r"\bbana\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)\s+de",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().title()
    return None


def format_memory_context(chat_id: int) -> str:
    memory = get_chat_memory(chat_id)
    lines = []
    if memory.get("name"):
        lines.append(f"Kullanicinin kayitli adi: {memory['name']}")
    else:
        lines.append("Kullanicinin kayitli adi henuz yok.")
    history = memory.get("history", [])
    if history:
        lines.append("Son konusma gecmisi:")
        lines.extend(history[-MAX_MEMORY_TURNS:])
    return "\n".join(lines)


def remember_exchange(chat_id: int, user_text: str, assistant_text: str) -> None:
    memory = get_chat_memory(chat_id)
    history = memory["history"]
    history.append(f"Kullanici: {user_text}")
    history.append(f"Asistan: {assistant_text}")
    memory["history"] = history[-(MAX_MEMORY_TURNS * 2) :]

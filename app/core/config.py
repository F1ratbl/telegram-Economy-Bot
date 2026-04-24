import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(".env")
load_dotenv(".ENV")


def get_env(*names: str, required: bool = False) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    if required:
        raise RuntimeError(f"Eksik ortam degiskeni: {', '.join(names)}")
    return None


TELEGRAM_BOT_TOKEN = get_env(
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_TOKEN",
    "BOT_TOKEN",
    "HTTP_API",
    required=True,
)
GOOGLE_API_KEY = get_env(
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_STUDIO_API",
    required=True,
)
WEBHOOK_BASE_URL = get_env("WEBHOOK_URL", "NGROK_URL", "WEBHOOK_BASE_URL")
GEMINI_MODEL_NAME = get_env("GEMINI_MODEL", "GEMINI_MODEL_NAME") or "gemini-2.5-flash"
ALPHA_VANTAGE_API_KEY = get_env("ALPHA_VANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY", "AV_API_KEY")
IS_VERCEL = bool(os.getenv("VERCEL"))
VOICE_ENABLED = (get_env("VOICE_ENABLED") or ("false" if IS_VERCEL else "true")).lower() in {"1", "true", "yes", "on"}

MAX_OUTPUT_TOKENS = 400
MAX_MEMORY_TURNS = 8
KB_COLLECTION_NAME = get_env("KB_COLLECTION_NAME") or "us_stock_market_knowledge"
KB_DB_DIR = Path(get_env("KB_DB_DIR") or "knowledge_base/chroma_db")
KB_TOP_K = 4
UNKNOWN_MESSAGE = "Bu konuda bilgim yok, bilgi bulamadım."
MARKET_TOOL_UNAVAILABLE_MESSAGE = "Canli piyasa verisi araci su anda hazir degil. API anahtarini kontrol et."
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
TELEGRAM_FILE_BASE = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}"
TEMP_AUDIO_DIR = Path(get_env("TEMP_AUDIO_DIR") or ("/tmp/telegram_economy_audio" if IS_VERCEL else "tmp_audio"))
TEMP_AUDIO_DIR.mkdir(exist_ok=True)
if not KB_DB_DIR.exists():
    KB_DB_DIR.mkdir(parents=True, exist_ok=True)

NON_US_MARKET_KEYWORDS = {
    "altin",
    "gold",
    "gumus",
    "silver",
    "japon borsasi",
    "japan stock",
    "nikkei",
    "dolar",
    "usdtry",
    "usd/try",
    "euro",
    "eurtry",
    "eur/try",
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "kripto",
    "crypto",
    "bist",
    "borsa istanbul",
    "viop",
    "gram altin",
    "ons altin",
    "faizsiz altin",
}
US_STOCK_MARKET_KEYWORDS = {
    "abd borsasi",
    "abd borsalari",
    "amerikan borsasi",
    "amerika borsasi",
    "abd hisse",
    "abd hisse senedi",
    "us stock",
    "us stocks",
    "nasdaq",
    "nyse",
    "wall street",
    "dow jones",
    "s&p 500",
    "sp500",
    "s&p500",
    "sec",
    "market order",
    "limit order",
    "short selling",
    "short",
    "aciga satis",
    "aciga satıs",
    "margin account",
    "margin hesap",
    "marjin hesap",
    "cash account",
    "nakit hesap",
    "pattern day trader",
    "day trade",
    "gun ici islem",
    "after hours",
    "after-hours",
    "pre market",
    "pre-market",
    "us market",
    "amerikan hissesi",
    "abd hissesi",
    "brokerage account",
    "broker hesabi",
}
TOOL_FOREX_KEYWORDS = {
    "doviz",
    "doviz kuru",
    "kur",
    "usd",
    "eur",
    "try",
    "dolar kuru",
    "euro kuru",
    "usdtry",
    "usd/try",
    "eurtry",
    "eur/try",
    "dolar tl",
    "euro tl",
}
TOOL_INDEX_KEYWORDS = {
    "abd borsa endeksi",
    "abd endeksi",
    "endeks",
    "s&p 500",
    "sp500",
    "s&p500",
    "spx",
    "nasdaq",
    "nasdaq 100",
    "ndx",
    "dow jones",
    "dow",
    "dji",
}
TOOL_OIL_KEYWORDS = {
    "petrol",
    "ham petrol",
    "wti",
    "brent",
    "crude oil",
    "oil price",
}

SYSTEM_INSTRUCTION = """
Sen uzman bir finansal analistsin.
Kullanicinin ekonomi, piyasa, enflasyon, faiz, doviz, yatirim ve kisisel finans sorularini
anlasilir, profesyonel ve dikkatli bir dille yanitlarsin.
Yanitlarinda finansal riskleri abartmadan ama net sekilde belirt.
Kesinlik iddiasi tasiyan yatirim tavsiyeleri verme; gerektiginde belirsizlikleri soyle.
KRITIK KURAL: Her zaman ve sadece Turkce yanit ver.
Yanitlarinda markdown kullanma.
Yildiz sembolu kullanma.
Kullanicinin adi biliniyorsa dogal sekilde adiyla hitap et.
Yanitlarin robotik, kalip veya soguk durmasin; dogal, akici ve insan gibi yaz.
Gereksiz basliklar kullanma.
Mumkun oldugunda dogrudan sorunun cevabiyla basla.
Kisa ve dolu cumleler kur; bos tekrar yapma.
""".strip()

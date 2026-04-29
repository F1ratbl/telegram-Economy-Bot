import logging
import time

from app.core.config import (
    ALPHA_VANTAGE_API_KEY,
    ALPHA_VANTAGE_BASE_URL,
    MARKET_TOOL_UNAVAILABLE_MESSAGE,
    TOOL_FOREX_KEYWORDS,
    TOOL_INDEX_KEYWORDS,
    TOOL_OIL_KEYWORDS,
)
from app.core.perf import log_timing
from app.services.ai_service import verbalize_market_reply
from app.services.state import ALPHA_VANTAGE_LOCK, HTTP_CLIENT
from app.services.text_service import contains_keyword_variation, normalize_topic_text
import app.services.state as state


logger = logging.getLogger("economy-assistant-bot")


@log_timing()
def alpha_vantage_request(params: dict[str, str]) -> dict[str, object]:
    if not ALPHA_VANTAGE_API_KEY:
        raise RuntimeError(MARKET_TOOL_UNAVAILABLE_MESSAGE)

    with ALPHA_VANTAGE_LOCK:
        now = time.monotonic()
        wait_seconds = 1.2 - (now - state.LAST_ALPHA_VANTAGE_REQUEST_AT)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        request_params = {**params, "apikey": ALPHA_VANTAGE_API_KEY}
        response = HTTP_CLIENT.get(ALPHA_VANTAGE_BASE_URL, params=request_params)
        state.LAST_ALPHA_VANTAGE_REQUEST_AT = time.monotonic()

    response.raise_for_status()
    payload = response.json()
    if "Error Message" in payload:
        raise RuntimeError(payload["Error Message"])
    if "Information" in payload:
        raise RuntimeError(payload["Information"])
    if "Note" in payload:
        raise RuntimeError(payload["Note"])
    return payload


def detect_market_tool_intent(user_text: str) -> str | None:
    normalized = normalize_topic_text(user_text)
    if any(contains_keyword_variation(normalized, keyword) for keyword in TOOL_OIL_KEYWORDS):
        return "oil"
    if any(contains_keyword_variation(normalized, keyword) for keyword in TOOL_INDEX_KEYWORDS):
        return "index"
    if any(contains_keyword_variation(normalized, keyword) for keyword in TOOL_FOREX_KEYWORDS):
        return "forex"
    return None


def detect_forex_pair(user_text: str) -> tuple[str, str] | None:
    normalized = normalize_topic_text(user_text)
    if any(token in normalized for token in {"eur", "euro"}):
        return ("EUR", "TRY")
    if any(token in normalized for token in {"usd", "dolar", "dollar"}):
        return ("USD", "TRY")
    return None


def detect_index_symbol(user_text: str) -> tuple[str, str]:
    normalized = normalize_topic_text(user_text)
    if any(token in normalized for token in {"s&p 500", "sp500", "s&p500", "spx"}):
        return ("SPX", "S&P 500")
    if any(token in normalized for token in {"nasdaq 100", "nasdaq", "ndx"}):
        return ("NDX", "Nasdaq 100")
    if any(token in normalized for token in {"dow jones", "dow", "dji"}):
        return ("DJI", "Dow Jones")
    return ("SPX", "S&P 500")


def get_index_proxy_symbol(index_symbol: str) -> tuple[str, str]:
    proxy_map = {
        "SPX": ("SPY", "S&P 500 ETF proxy (SPY)"),
        "NDX": ("QQQ", "Nasdaq 100 ETF proxy (QQQ)"),
        "DJI": ("DIA", "Dow Jones ETF proxy (DIA)"),
    }
    return proxy_map.get(index_symbol, ("SPY", "S&P 500 ETF proxy (SPY)"))


def parse_latest_oil_value(payload: dict[str, object]) -> tuple[str, str]:
    data = payload.get("data") or []
    if not data:
        raise RuntimeError("Petrol verisi bulunamadi.")
    latest = data[0]
    return str(latest.get("date", "-")), str(latest.get("value", "-"))


def parse_global_quote(payload: dict[str, object]) -> tuple[str, str]:
    data = payload.get("Global Quote") or {}
    price = data.get("05. price")
    trading_day = data.get("07. latest trading day", "-")
    if not price:
        raise RuntimeError("Global quote verisi bulunamadi.")
    return trading_day, price


def _looks_like_direct_price_question(user_text: str) -> bool:
    normalized = normalize_topic_text(user_text)
    direct_patterns = ["kac", "ne kadar", "fiyat", "fiyati", "kacti", "guncel"]
    return any(pattern in normalized for pattern in direct_patterns)


@log_timing()
def get_forex_rate_reply(user_text: str) -> str:
    pair = detect_forex_pair(user_text)
    if not pair:
        return "Hangi kuru istedigini anlamadim. Ornek: dolar kuru veya euro kuru yazabilirsin."
    from_currency, to_currency = pair
    payload = alpha_vantage_request(
        {"function": "CURRENCY_EXCHANGE_RATE", "from_currency": from_currency, "to_currency": to_currency}
    )
    data = payload.get("Realtime Currency Exchange Rate") or {}
    rate = data.get("5. Exchange Rate")
    last_refreshed = data.get("6. Last Refreshed", "-")
    if not rate:
        raise RuntimeError("Doviz kuru verisi bulunamadi.")
    facts = {
        "varlik": f"{from_currency}/{to_currency}",
        "guncel_seviye": str(rate),
        "son_guncellenme": str(last_refreshed),
        "kaynak": "Alpha Vantage",
    }
    try:
        return verbalize_market_reply(user_text, facts)
    except Exception:
        logger.exception("Doviz verisi dogal dile cevrilemedi, sabit metne donuluyor.")
    if _looks_like_direct_price_question(user_text):
        return f"Su an {from_currency}/{to_currency} kuru {rate} seviyesinde gorunuyor. Son guncellenme: {last_refreshed}."
    return (
        f"{from_currency}/{to_currency} tarafinda guncel seviye {rate}. "
        f"Bu veri {last_refreshed} zaman damgasiyla geldi."
    )


@log_timing()
def get_us_index_reply(user_text: str) -> str:
    symbol, label = detect_index_symbol(user_text)
    proxy_symbol, proxy_label = get_index_proxy_symbol(symbol)
    payload = alpha_vantage_request({"function": "GLOBAL_QUOTE", "symbol": proxy_symbol})
    latest_date, latest_value = parse_global_quote(payload)
    facts = {
        "varlik": label,
        "referans_sembol": proxy_symbol,
        "referans_tanim": proxy_label,
        "guncel_seviye": str(latest_value),
        "veri_tarihi": str(latest_date),
        "not": "Bu veri birebir resmi endeks seviyesi degil, ona yakin bir referanstir.",
        "kaynak": "Alpha Vantage",
    }
    try:
        return verbalize_market_reply(user_text, facts)
    except Exception:
        logger.exception("Endeks verisi dogal dile cevrilemedi, sabit metne donuluyor.")
    if _looks_like_direct_price_question(user_text):
        short_label_map = {
            "Nasdaq 100": "Nasdaq tarafi",
            "S&P 500": "S&P 500 tarafi",
            "Dow Jones": "Dow Jones tarafi",
        }
        natural_label = short_label_map.get(label, label)
        return (
            f"Su an {natural_label} icin elimdeki en yakin gosterge {latest_value}. "
            f"Bunu {proxy_symbol} uzerinden takip ediyorum; yani bu birebir resmi endeks seviyesi degil, ona yakin bir referans. "
            f"Veri tarihi de {latest_date}."
        )
    return (
        f"{label} icin ucretsiz veri siniri nedeniyle {proxy_label} referans alindi. "
        f"Guncel seviye {latest_value}, veri tarihi ise {latest_date}. "
        f"Bunu endeksin birebir resmi seviyesi degil, ona yakin bir piyasa gostergesi gibi dusunebilirsin."
    )


@log_timing()
def get_oil_price_reply() -> str:
    payload = alpha_vantage_request({"function": "WTI", "interval": "daily"})
    latest_date, latest_value = parse_latest_oil_value(payload)
    facts = {
        "varlik": "WTI ham petrol",
        "guncel_seviye": str(latest_value),
        "veri_tarihi": str(latest_date),
        "birim": "USD",
        "kaynak": "Alpha Vantage",
    }
    try:
        return verbalize_market_reply("petrol fiyatı", facts)
    except Exception:
        logger.exception("Petrol verisi dogal dile cevrilemedi, sabit metne donuluyor.")
    return (
        f"WTI ham petrol tarafinda guncel seviye {latest_value} USD. "
        f"Veri tarihi {latest_date}."
    )


@log_timing()
def answer_with_market_tool(user_text: str) -> str | None:
    intent = detect_market_tool_intent(user_text)
    if intent is None:
        return None
    try:
        if intent == "forex":
            return get_forex_rate_reply(user_text)
        if intent == "index":
            return get_us_index_reply(user_text)
        if intent == "oil":
            return get_oil_price_reply()
    except RuntimeError as exc:
        logger.warning("Market tool hatasi: %s", exc)
        return f"Canli veri cekilemedi: {exc}"
    return None

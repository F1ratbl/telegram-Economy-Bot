import re


def sanitize_reply_text(text: str) -> str:
    cleaned = text.replace("*", "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_topic_text(text: str) -> str:
    lowered = text.lower()
    replacements = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"})
    return lowered.translate(replacements)


def contains_keyword_variation(text: str, keyword: str) -> bool:
    if keyword in text:
        return True
    compact_text = re.sub(r"[^a-z0-9]+", " ", text)
    compact_keyword = re.sub(r"[^a-z0-9]+", " ", keyword)
    return compact_keyword in compact_text


def build_kb_search_queries(text: str) -> list[str]:
    queries: list[str] = []
    base_text = text.strip()
    if base_text:
        queries.append(base_text)

    simplified = normalize_topic_text(base_text)
    simplified = re.sub(
        r"\b(fiyati|fiyat|kac|kac oldu|son durum|guncel|gunceli|bugun|su an|anlik|ne kadar)\b",
        " ",
        simplified,
    )
    simplified = re.sub(r"\s+", " ", simplified).strip()
    if simplified and simplified not in queries:
        queries.append(simplified)

    return queries


def is_capability_question(text: str) -> bool:
    normalized = normalize_topic_text(text)
    patterns = [
        "neyi biliyorsun",
        "neler biliyorsun",
        "ne biliyorsun",
        "hangi konulari biliyorsun",
        "hangi konularda yardimci olabilirsin",
        "ne yapabiliyorsun",
    ]
    return any(pattern in normalized for pattern in patterns)


def is_greeting_question(text: str) -> bool:
    normalized = normalize_topic_text(text)
    patterns = [
        "merhaba",
        "selam",
        "selamlar",
        "gunaydin",
        "iyi gunler",
        "iyi aksamlar",
        "iyi geceler",
        "hos bulduk",
        "hosbuldum",
    ]
    return any(pattern in normalized for pattern in patterns)


def is_how_are_you_question(text: str) -> bool:
    normalized = normalize_topic_text(text)
    patterns = [
        "nasilsin",
        "naber",
        "napiyorsun",
    ]
    return any(pattern in normalized for pattern in patterns)


def is_smalltalk_question(text: str) -> bool:
    return is_greeting_question(text) or is_how_are_you_question(text)


def is_general_economy_question(text: str) -> bool:
    normalized = normalize_topic_text(text)
    keywords = [
        "ekonomi",
        "enflasyon",
        "faiz",
        "resesyon",
        "stagflasyon",
        "issizlik",
        "merkez bankasi",
        "fed",
        "tcmb",
        "doviz",
        "kur",
        "dolar",
        "euro",
        "petrol",
        "emtia",
        "altin",
        "gumus",
        "borsa",
        "endeks",
        "hisse",
        "kripto",
        "bitcoin",
        "ethereum",
        "buyume",
        "cari acik",
        "butce acigi",
        "tahvil",
        "bono",
        "parite",
    ]
    return any(contains_keyword_variation(normalized, keyword) for keyword in keywords)


def is_asking_stored_name(text: str) -> bool:
    normalized = normalize_topic_text(text)
    patterns = [
        "benim adim ne",
        "adim ne",
        "ismim ne",
        "ben kimim",
    ]
    return any(pattern in normalized for pattern in patterns)

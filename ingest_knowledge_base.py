import argparse
import os
import re
import uuid
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
import chromadb
from pypdf import PdfReader


load_dotenv(".env")
load_dotenv(".ENV")

KB_COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "us_stock_market_knowledge")
KB_DB_DIR = Path(os.getenv("KB_DB_DIR", "knowledge_base/chroma_db"))
RAW_DOCS_DIR = Path(os.getenv("KB_RAW_DOCS_DIR", "knowledge_base/raw"))


def iter_source_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
            yield path


def read_file_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8").strip()

    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("PDF okumak icin `pip install pypdf` gerekiyor.")

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    raise RuntimeError(f"Desteklenmeyen dosya tipi: {path.suffix}")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    if len(paragraph) <= chunk_size:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            parts.append(current)
        if len(sentence) <= chunk_size:
            current = sentence
            continue

        start = 0
        while start < len(sentence):
            end = min(start + chunk_size, len(sentence))
            piece = sentence[start:end].strip()
            if piece:
                parts.append(piece)
            start = end
        current = ""

    if current:
        parts.append(current)

    return parts


def chunk_text(text: str, chunk_size: int = 1200) -> list[str]:
    clean_text = normalize_whitespace(text)
    if not clean_text:
        return []

    paragraphs = [paragraph.strip() for paragraph in clean_text.split("\n\n") if paragraph.strip()]
    normalized_paragraphs: list[str] = []
    for paragraph in paragraphs:
        normalized_paragraphs.extend(split_long_paragraph(paragraph, chunk_size))

    chunks: list[str] = []
    current = ""

    for paragraph in normalized_paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current.strip())
        current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def ingest_documents(input_dir: Path, reset: bool = False) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    KB_DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(KB_DB_DIR))

    if reset:
        try:
            client.delete_collection(KB_COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=KB_COLLECTION_NAME)

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict[str, str | int]] = []

    for file_path in iter_source_files(input_dir):
        text = read_file_text(file_path)
        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{file_path.stem}-{index}-{uuid.uuid4().hex[:8]}")
            all_metadatas.append(
                {
                    "source": str(file_path),
                    "chunk_index": index,
                    "topic": "ABD Borsasi Islemleri",
                }
            )

    if not all_chunks:
        raise SystemExit(
            f"Kaydedilecek dokuman bulunamadi. ABD Borsasi dokumanlarini {input_dir} altina ekle."
        )

    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadatas,
    )

    print(f"{len(all_chunks)} chunk ChromaDB icine kaydedildi.")
    print(f"Collection: {KB_COLLECTION_NAME}")
    print(f"DB Path: {KB_DB_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ABD Borsasi dokumanlarini chunk edip ChromaDB knowledgebase olarak kaydeder."
    )
    parser.add_argument(
        "--input-dir",
        default=str(RAW_DOCS_DIR),
        help="Ham dokumanlarin bulundugu klasor",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Var olan collection'i silip sifirdan kur",
    )
    args = parser.parse_args()

    ingest_documents(Path(args.input_dir), reset=args.reset)


if __name__ == "__main__":
    main()

import argparse
import os
import re
import uuid
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
import google.generativeai as genai
from pypdf import PdfReader
from qdrant_client import QdrantClient, models


load_dotenv(".env")
load_dotenv(".ENV")

KB_COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "us_stock_market_knowledge")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", KB_COLLECTION_NAME)
KB_EMBEDDING_MODEL = os.getenv("KB_EMBEDDING_MODEL", "models/text-embedding-004")
RAW_DOCS_DIR = Path(os.getenv("KB_RAW_DOCS_DIR", "knowledge_base/raw"))
GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_STUDIO_API")
)

genai.configure(api_key=GOOGLE_API_KEY)


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


def embed_document_text(text: str) -> list[float]:
    response = genai.embed_content(
        model=KB_EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document",
    )
    embedding = response.get("embedding") or []
    return [float(value) for value in embedding]


def ingest_documents(input_dir: Path, reset: bool = False) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    if not GOOGLE_API_KEY:
        raise SystemExit("Qdrant ingestion icin GOOGLE_API_KEY veya GOOGLE_STUDIO_API gerekiyor.")
    if not QDRANT_URL:
        raise SystemExit("Qdrant ingestion icin QDRANT_URL gerekli.")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0)

    all_chunks: list[str] = []
    all_points: list[models.PointStruct] = []

    for file_path in iter_source_files(input_dir):
        text = read_file_text(file_path)
        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            embedding = embed_document_text(chunk)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path}:{index}:{chunk}"))
            all_points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document": chunk,
                        "source": str(file_path),
                        "chunk_index": index,
                        "topic": "ABD Borsasi Islemleri",
                    },
                )
            )

    if not all_chunks:
        raise SystemExit(
            f"Kaydedilecek dokuman bulunamadi. ABD Borsasi dokumanlarini {input_dir} altina ekle."
        )

    vector_size = len(all_points[0].vector)
    if reset:
        try:
            client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
        except Exception:
            pass

    try:
        client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=all_points)

    print(f"{len(all_chunks)} chunk Qdrant icine kaydedildi.")
    print(f"Collection: {QDRANT_COLLECTION_NAME}")
    print(f"Qdrant URL: {QDRANT_URL}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ABD Borsasi dokumanlarini chunk edip Qdrant knowledgebase olarak kaydeder."
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

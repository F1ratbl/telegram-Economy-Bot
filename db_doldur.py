from ingest_knowledge_base import ingest_documents, RAW_DOCS_DIR


if __name__ == "__main__":
    ingest_documents(RAW_DOCS_DIR, reset=True)

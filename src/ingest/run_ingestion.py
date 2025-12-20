import json
from pathlib import Path
from tqdm import tqdm

from src.config import (
    WHO_DIR,
    CDC_DIR,
    CHUNK_OUTPUT_FILE,
    CHUNK_WORDS,
    CHUNK_OVERLAP,
)
from src.ingest.pdf_loader import extract_pdf_pages
from src.ingest.text_cleaner import clean_text
from src.ingest.chunker import chunk_pages


def ingest_source(source_dir: Path, source_name: str):
    all_chunks = []

    for pdf_file in tqdm(list(source_dir.glob("*.pdf")), desc=f"Ingesting {source_name}"):
        pages = extract_pdf_pages(pdf_file)

        # clean text per page
        for page in pages:
            page["text"] = clean_text(page["text"])

        chunks = chunk_pages(
            pages,
            chunk_words=CHUNK_WORDS,
            overlap=CHUNK_OVERLAP
        )

        for chunk in chunks:
            chunk.update({
                "source": source_name,
                "document": pdf_file.name
            })
            all_chunks.append(chunk)

    return all_chunks


def main():
    CHUNK_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    all_chunks.extend(ingest_source(WHO_DIR, "WHO"))
    all_chunks.extend(ingest_source(CDC_DIR, "CDC"))

    with open(CHUNK_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nIngestion complete.")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Saved to: {CHUNK_OUTPUT_FILE}")


if __name__ == "__main__":
    main()

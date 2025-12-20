import uuid
from typing import List, Dict

def chunk_pages(
    pages: List[Dict],
    chunk_words: int,
    overlap: int
) -> List[Dict]:
    """
    Chunk text while preserving page ranges.
    """
    chunks = []

    buffer_words = []
    buffer_pages = []

    for page in pages:
        words = page["text"].split()
        buffer_words.extend(words)
        buffer_pages.append(page["page_number"])

        while len(buffer_words) >= chunk_words:
            chunk_text = " ".join(buffer_words[:chunk_words])

            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text,
                "page_start": min(buffer_pages),
                "page_end": max(buffer_pages)
            })

            buffer_words = buffer_words[chunk_words - overlap:]
            buffer_pages = buffer_pages[-1:]

    # last chunk
    if buffer_words:
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": " ".join(buffer_words),
            "page_start": min(buffer_pages),
            "page_end": max(buffer_pages)
        })

    return chunks

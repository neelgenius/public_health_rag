from pathlib import Path

from src.config import CHUNK_OUTPUT_FILE, PROCESSED_DIR
from src.indexing.load_corpus import load_chunks
from src.indexing.vector_index import VectorIndex
from src.indexing.bm25_index import BM25Index


VECTOR_INDEX_PATH = PROCESSED_DIR / "vector.faiss"
VECTOR_META_PATH = PROCESSED_DIR / "vector_meta.json"
BM25_PATH = PROCESSED_DIR / "bm25.pkl"


def main():
    chunks = load_chunks(CHUNK_OUTPUT_FILE)
    print(f"Loaded {len(chunks)} chunks")

    # Vector index
    print("Building vector index...")
    vector_index = VectorIndex()
    vector_index.build(chunks)
    vector_index.save(VECTOR_INDEX_PATH, VECTOR_META_PATH)

    # BM25 index
    print("Building BM25 index...")
    bm25_index = BM25Index()
    bm25_index.build(chunks)
    bm25_index.save(BM25_PATH)

    print("\nIndexing complete.")
    print(f"Vector index saved to: {VECTOR_INDEX_PATH}")
    print(f"BM25 index saved to: {BM25_PATH}")


if __name__ == "__main__":
    main()

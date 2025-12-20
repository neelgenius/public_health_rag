import json
from pathlib import Path

from src.config import PROCESSED_DIR, CHUNK_OUTPUT_FILE
from src.retrieval.vector_search import VectorSearcher
from src.retrieval.bm25_search import BM25Searcher
from src.indexing.fusion import reciprocal_rank_fusion


VECTOR_INDEX = PROCESSED_DIR / "vector.faiss"
VECTOR_META = PROCESSED_DIR / "vector_meta.json"
BM25_INDEX = PROCESSED_DIR / "bm25.pkl"


class HybridRetriever:
    def __init__(self):
        self.vector_searcher = VectorSearcher(VECTOR_INDEX, VECTOR_META)
        self.bm25_searcher = BM25Searcher(BM25_INDEX)
        self.chunks = self._load_chunks()

    def _load_chunks(self):
        chunks = {}
        with open(CHUNK_OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                chunks[c["chunk_id"]] = c
        return chunks

    def retrieve(self, query: str, top_k: int = 5):
        vector_results = self.vector_searcher.search(query, top_k=20)
        bm25_results = self.bm25_searcher.search(query, top_k=20)

        rrf_scores = reciprocal_rank_fusion([
            [r["chunk_id"] for r in vector_results],
            [r["chunk_id"] for r in bm25_results]
        ])

        combined = []
        for chunk_id, fused_score in rrf_scores.items():
            c = self.chunks[chunk_id]

            combined.append({
                "chunk_id": chunk_id,
                "text": c["text"],
                "source": c["source"],
                "document": c["document"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "fused_score": fused_score,
                "vector_score": next(
                    (r["score"] for r in vector_results if r["chunk_id"] == chunk_id),
                    0.0
                ),
                "bm25_score": next(
                    (r["score"] for r in bm25_results if r["chunk_id"] == chunk_id),
                    0.0
                ),
            })

        combined.sort(key=lambda x: x["fused_score"], reverse=True)
        return combined[:top_k]

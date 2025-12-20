import pickle
from pathlib import Path


class BM25Searcher:
    def __init__(self, bm25_path: Path):
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]

    def search(self, query: str, top_k: int = 10):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [
            {
                "chunk_id": self.chunk_ids[i],
                "score": float(score)
            }
            for i, score in ranked
        ]

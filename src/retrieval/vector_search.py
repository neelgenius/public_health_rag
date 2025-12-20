import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


class VectorSearcher:
    def __init__(self, index_path: Path, meta_path: Path):
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def search(self, query: str, top_k: int = 10):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "chunk_id": self.id_map[str(idx)],
                "score": float(score)
            })

        return results

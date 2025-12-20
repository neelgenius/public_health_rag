import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class VectorIndex:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384
    ):
        self.model = SentenceTransformer(model_name)
        self.dim = dim

        # Inner Product index (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dim)

        # maps internal FAISS ids -> chunk_id
        self.id_map = {}

    def build(self, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        self.id_map = {
            i: chunks[i]["chunk_id"] for i in range(len(chunks))
        }

    def save(self, index_path: Path, meta_path: Path):
        faiss.write_index(self.index, str(index_path))

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f)

    def load(self, index_path: Path, meta_path: Path):
        self.index = faiss.read_index(str(index_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

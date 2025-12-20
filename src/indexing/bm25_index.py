import pickle
from pathlib import Path
from typing import List, Dict
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunk_ids = []

    def build(self, chunks: List[Dict]):
        tokenized_corpus = [
            c["text"].lower().split() for c in chunks
        ]

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunk_ids": self.chunk_ids
            }, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunk_ids = data["chunk_ids"]

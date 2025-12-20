import json
from pathlib import Path
from typing import List, Dict


def load_chunks(path: Path) -> List[Dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

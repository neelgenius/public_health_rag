from collections import defaultdict
from typing import Dict, List


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60
) -> Dict[str, float]:
    """
    ranked_lists: list of ranked chunk_id lists
    returns: chunk_id -> fused score
    """
    scores = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            scores[chunk_id] += 1.0 / (k + rank + 1)

    return dict(scores)

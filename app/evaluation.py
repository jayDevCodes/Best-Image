"""Offline ranking metrics for user-labelled image comparisons.

These metrics evaluate a frozen candidate ordering against human labels. Use a held-out
JSONL file for comparison; measuring on the same examples used for preference learning
will overstate generalization.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def _ranking_indices(raw_ranking: Any) -> list[int]:
    if not isinstance(raw_ranking, list) or not raw_ranking:
        raise ValueError("Each case needs a non-empty ranking list")
    indices: list[int] = []
    for item in raw_ranking:
        if isinstance(item, Mapping):
            if "index" not in item:
                raise ValueError("Each ranking item must contain an index")
            value = item["index"]
        else:
            value = item
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Ranking indices must be integers")
        indices.append(value)
    if len(set(indices)) != len(indices):
        raise ValueError("Ranking indices must be unique within a comparison")
    return indices


def evaluate_rankings(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute top-1/top-3 accuracy, reciprocal rank and mean preferred rank.

    Each case has a ranking (candidate indices or ranker candidate objects) and
    preferred_index (the human-confirmed winner). No prompts or image bytes are read.
    """
    total = 0
    top1 = 0
    top3 = 0
    reciprocal_rank_sum = 0.0
    preferred_rank_sum = 0.0
    histogram: dict[str, int] = {}

    for number, case in enumerate(cases, start=1):
        if not isinstance(case, Mapping):
            raise ValueError(f"Case {number} must be an object")
        indices = _ranking_indices(case.get("ranking"))
        preferred = case.get("preferred_index")
        if isinstance(preferred, bool) or not isinstance(preferred, int):
            raise ValueError(f"Case {number} needs an integer preferred_index")
        if preferred not in indices:
            raise ValueError(f"Case {number}: preferred_index is absent from the ranking")
        rank = indices.index(preferred) + 1
        total += 1
        top1 += int(rank == 1)
        top3 += int(rank <= 3)
        reciprocal_rank_sum += 1.0 / rank
        preferred_rank_sum += rank
        key = str(rank)
        histogram[key] = histogram.get(key, 0) + 1

    if total == 0:
        raise ValueError("The benchmark must contain at least one labelled comparison")

    return {
        "comparison_count": total,
        "top1_accuracy": round(top1 / total, 4),
        "top3_accuracy": round(top3 / total, 4),
        "mean_reciprocal_rank": round(reciprocal_rank_sum / total, 4),
        "mean_preferred_rank": round(preferred_rank_sum / total, 4),
        "preferred_rank_histogram": {key: histogram[key] for key in sorted(histogram, key=int)},
        "warning": "These metrics are meaningful only when the labels are representative and the evaluation set is held out from tuning.",
    }

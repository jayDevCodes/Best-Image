"""Conservative prompt-clause extraction and review warnings."""
from __future__ import annotations
import re

MAX_REQUIREMENTS = 5

def extract_requirements(prompt: str) -> list[str]:
    """Split only on obvious boundaries; this is not a semantic parser."""
    raw_parts = re.split(r"[;,\n]+|(?<=[.!?])\s+", prompt.strip())
    seen: set[str] = set()
    parts: list[str] = []
    for part in raw_parts:
        normalized = re.sub(r"\s+", " ", part).strip(" .,:;\t")
        key = normalized.casefold()
        if len(normalized) >= 5 and key not in seen:
            parts.append(normalized)
            seen.add(key)
        if len(parts) >= MAX_REQUIREMENTS:
            break
    return parts if len(parts) > 1 else []

def review_warnings(prompt: str) -> list[str]:
    """Warn about requirements that the baseline cannot independently verify."""
    warnings: list[str] = []
    if re.search(r"\b(exactly|at least|at most|no more than|number of|one|two|three|four|five|six|\d+)\b", prompt, re.IGNORECASE):
        warnings.append("The prompt may require exact counting. v0.1 uses image-text similarity, not a dedicated object counter.")
    if re.search(r"\b(text|spell|spelling|lettering|logo|sign|caption|written|words?)\b", prompt, re.IGNORECASE):
        warnings.append("The prompt mentions text or lettering. Exact OCR and spelling verification are not included in v0.1.")
    if re.search(r"\b(left|right|behind|in front of|next to|above|below|beside|near|foreground|background|between)\b", prompt, re.IGNORECASE):
        warnings.append("The prompt contains a spatial relation. Similarity scoring may miss subtle placement errors.")
    if not warnings:
        warnings.append("Scores are similarity and image-statistic signals, not proof that every visual detail is correct.")
    return warnings

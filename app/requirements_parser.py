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

def review_warnings(
    prompt: str,
    *,
    ocr_available: bool = False,
    ocr_scored: bool = False,
    object_scored: bool = False,
    spatial_scored: bool = False,
) -> list[str]:
    """Warn precisely about visual checks used and the remaining capability gaps."""
    warnings: list[str] = []
    count_requested = re.search(
        r"\b(exactly|at least|at most|no more than|number of|one|two|three|four|five|six|\d+)\b",
        prompt, re.IGNORECASE,
    )
    text_requested = re.search(
        r"\b(text|spell|spelling|lettering|logo|sign|caption|written|words?|says?|read(?:s|ing)?|printed)\b",
        prompt, re.IGNORECASE,
    )
    spatial_requested = re.search(
        r"\b(left|right|behind|in front of|next to|above|below|under|beside|near|foreground|background|between)\b",
        prompt, re.IGNORECASE,
    )

    if count_requested:
        if object_scored:
            warnings.append(
                "An optional COCO object detector contributed a count/presence signal for supported classes. "
                "It may miss small, occluded, stylized, or unsupported objects; verify exact counts manually."
            )
        else:
            warnings.append(
                "The prompt may require exact counting. The current analysis did not use an object-count signal; verify counts manually."
            )

    if text_requested:
        if ocr_scored:
            warnings.append(
                "Optional local OCR compared quoted prompt text. OCR can misread stylized, small, or distorted letters; verify spelling visually."
            )
        elif ocr_available:
            warnings.append(
                "Local OCR is available, but no exact quoted phrase could be scored for this prompt; verify text and spelling manually."
            )
        else:
            warnings.append(
                "The prompt mentions text or lettering, but local OCR is unavailable; verify exact spelling visually."
            )

    if spatial_requested:
        if spatial_scored:
            warnings.append(
                "An optional detector estimated simple left/right/above/below relations from 2-D box centers. "
                "It does not reliably verify depth relations such as 'behind' or 'in front of'."
            )
        else:
            warnings.append(
                "The prompt contains a spatial relation, but no spatial detector signal was used. Similarity scoring may miss placement errors."
            )

    if not warnings:
        warnings.append(
            "Scores are comparative visual signals, not proof that every prompt detail is correct. Inspect the top candidates."
        )
    return warnings

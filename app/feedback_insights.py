"""Lightweight labels for human feedback rationales.

These tags make recurring capability gaps measurable without pretending that free text
has changed the model weights. The current ranker learns from the chosen candidate and
its existing criterion scores; these tags are diagnostic metadata.
"""
from __future__ import annotations

import re

CATEGORY_LABELS = {
    "prompt_match": "Prompt / subject match",
    "visual_clarity": "Sharpness / visual clarity",
    "resolution_quality": "Resolution / technical quality",
    "text_or_spelling": "Text / spelling verification (not currently implemented)",
    "object_counting": "Exact object counting (not currently implemented)",
    "spatial_relations": "Spatial relations (not reliably verified)",
    "composition": "Composition / framing",
    "other": "Other / general feedback",
}

_PATTERNS = (
    ("text_or_spelling", re.compile(r"\b(text|spell(?:ing)?|lettering|letters|words?|ocr|caption|sign|logo|written)\b", re.I)),
    ("object_counting", re.compile(r"\b(count(?:ing)?|number of|exactly|at least|at most|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", re.I)),
    ("spatial_relations", re.compile(r"\b(left|right|behind|in front of|next to|beside|above|below|between|foreground|background|overlapping)\b", re.I)),
    ("visual_clarity", re.compile(r"\b(sharp(?:ness)?|blurr?y|blurred|focus|clear(?:er|ness)?|noise|artifact|grainy|distorted)\b", re.I)),
    ("resolution_quality", re.compile(r"\b(resolution|low.res|high.res|pixelated|pixels|4k|hd|dimensions|quality)\b", re.I)),
    ("composition", re.compile(r"\b(composition|framing|crop(?:ped|ping)?|lighting|angle|balance|aesthetic|layout|background)\b", re.I)),
    ("prompt_match", re.compile(r"\b(prompt|instruction|subject|matches?|requested|detail|color|colour|style|object|accurate|accuracy)\b", re.I)),
)


def tag_feedback_reason(reason: str) -> list[str]:
    """Return stable category IDs from user-provided feedback, without storing images/prompts."""
    text = " ".join((reason or "").split())[:2000]
    if not text:
        return []
    tags = [tag for tag, pattern in _PATTERNS if pattern.search(text)]
    return tags or ["other"]

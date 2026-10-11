"""Combine model relevance signals and image diagnostics into an explainable rank."""
from __future__ import annotations
from typing import Any
from app.config import normalized_weights

CRITERIA = {
    "prompt_match": "Prompt match",
    "requirement_match": "Inferred detail match",
    "image_clarity": "Edge/contrast heuristic",
    "technical_quality": "Resolution adequacy",
    "text_match": "Quoted-text OCR match (optional)",
    "object_match": "Object presence/count (optional detector)",
    "spatial_match": "2-D left/right/above/below (optional detector)",
    "preference_match": "Human-preference reward (optional ImageReward)",
    "vlm_match": "Optional local VLM detail review (heuristic)",
}

def rank_candidates(candidates: list[dict[str, Any]], weights: dict[str, float], requirement_labels: list[str]) -> list[dict[str, Any]]:
    """Attach normalized scores and deterministic, inspectable explanations."""
    if len(candidates) < 2:
        raise ValueError("At least two candidates are required")
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        diagnostics = candidate["diagnostics"]
        details = candidate["requirement_scores"]
        scores: dict[str, float] = {
            "prompt_match": float(candidate["prompt_match"]),
            "image_clarity": float(diagnostics["clarity_score"]),
            "technical_quality": float(diagnostics["resolution_score"]),
        }
        if details:
            scores["requirement_match"] = sum(details) / len(details)
        for optional_signal in ("text_match", "object_match", "spatial_match", "preference_match", "vlm_match"):
            if candidate.get(optional_signal) is not None:
                scores[optional_signal] = max(0.0, min(100.0, float(candidate[optional_signal])))
        active_weights = normalized_weights(weights, set(scores))
        final_score = sum(scores[key] * weight for key, weight in active_weights.items())
        reasons: list[str] = []
        if scores["prompt_match"] >= 65:
            reasons.append("Strong overall prompt-image similarity signal")
        elif scores["prompt_match"] < 40:
            reasons.append("Weak overall prompt-image similarity signal")
        if details:
            best, worst = max(details), min(details)
            if best - worst >= 15:
                reasons.append(f"Potentially weaker match for inferred detail: {requirement_labels[details.index(worst)]}")
            else:
                reasons.append("Inferred prompt-detail scores are relatively consistent")
        if diagnostics["clarity_score"] >= 60:
            reasons.append("Relatively strong edge clarity and contrast statistics")
        if diagnostics["resolution_score"] < 50:
            reasons.append("Low resolution relative to the app's 1-megapixel reference")
        if scores.get("text_match", 100.0) < 50:
            reasons.append("Optional quoted-text OCR found a weak literal match; inspect spelling manually")
        if scores.get("object_match", 100.0) < 50:
            reasons.append("Optional detector found a weak supported-object/count match; inspect counts manually")
        if scores.get("spatial_match", 100.0) < 50:
            reasons.append("Optional detector found a weak 2-D relation match; inspect placement manually")
        if scores.get("preference_match", 100.0) < 50:
            reasons.append("Optional ImageReward human-preference signal was low; inspect the image manually")
        if candidate.get("vlm_reason"):
            reasons.append("Unverified local VLM note: " + str(candidate["vlm_reason"])[:180])
        if not reasons:
            reasons.append("Review the score breakdown; no single heuristic dominates")
        ranked.append({
            "index": candidate["index"], "filename": candidate["filename"],
            "score": round(final_score, 2),
            "criteria": {
                key: {"label": CRITERIA[key], "score": round(value, 2), "weight": round(active_weights[key], 4)}
                for key, value in scores.items()
            },
            "requirement_scores": [
                {"requirement": requirement_labels[i], "score": round(value, 2)}
                for i, value in enumerate(details)
            ],
            "diagnostics": diagnostics, "reasons": reasons,
        })
    ranked.sort(key=lambda item: (-item["score"], -item["criteria"]["prompt_match"]["score"], item["index"]))
    for position, candidate in enumerate(ranked, start=1):
        candidate["rank"] = position
    gap = ranked[0]["score"] - ranked[1]["score"]
    # This is only the numerical separation between the top two candidates.
    ranked[0]["margin_band"] = "low" if gap < 3 else "medium" if gap < 8 else "high"
    ranked[0]["winner_summary"] = (
        f"Ranks first by the selected scoring profile with a {gap:.1f}-point gap to the next candidate. "
        "Treat this as a comparative recommendation, not a correctness probability."
    )
    return ranked

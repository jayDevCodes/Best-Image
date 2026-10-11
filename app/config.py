"""Load and validate user-editable scoring profiles."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "scoring_profiles.json"
DEFAULT_WEIGHTS = {
    "prompt_match": 0.46,
    "requirement_match": 0.22,
    "object_match": 0.08,
    "spatial_match": 0.04,
    "text_match": 0.06,
    "preference_match": 0.10,
    "image_clarity": 0.025,
    "technical_quality": 0.0132,
    "vlm_match": 0.12,
}

def load_profiles() -> dict[str, dict[str, Any]]:
    """Read scoring profiles and reject invalid weights instead of silently accepting them."""
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not read scoring profiles") from exc
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise RuntimeError("At least one scoring profile must be configured")
    validated: dict[str, dict[str, Any]] = {}
    required = set(DEFAULT_WEIGHTS)
    for key, profile in profiles.items():
        if not isinstance(key, str) or not isinstance(profile, dict):
            raise RuntimeError("Each profile must be an object with a string key")
        weights = profile.get("weights")
        if not isinstance(weights, dict) or set(weights) != required:
            raise RuntimeError(f"Profile {key!r} must define exactly: {sorted(required)}")
        numeric = {name: float(value) for name, value in weights.items()}
        if any(value < 0 or value > 1 for value in numeric.values()) or sum(numeric.values()) <= 0:
            raise RuntimeError(f"Profile {key!r} weights must be in [0, 1] and sum above zero")
        validated[key] = {
            "label": str(profile.get("label", key.replace("_", " ").title())),
            "description": str(profile.get("description", "")),
            "weights": numeric,
        }
    return validated

def normalized_weights(configured: dict[str, float], available_criteria: set[str]) -> dict[str, float]:
    """Normalize only criteria that exist for this analysis."""
    active = {key: value for key, value in configured.items() if key in available_criteria}
    total = sum(active.values())
    if total <= 0:
        raise ValueError("No scoring weights are available for this analysis")
    return {key: value / total for key, value in active.items()}

"""Opt-in local ImageReward human-preference signal.

The model is disabled by default because it adds a BLIP-Large backbone and substantial
memory/first-download cost. It only receives in-memory PIL images and never writes images.
"""
from __future__ import annotations

import importlib.util
import math
import os
import threading
from typing import Any

_ENABLED = os.getenv("BEST_IMAGE_ENABLE_PREFERENCE_REWARD", "").strip().lower() in {"1", "true", "yes", "on"}
_MODEL_NAME = "ImageReward-v1.0"
_MODEL: Any = None
_TORCH: Any = None
_DEVICE: str | None = None
_LOAD_ERROR: str | None = None
_LOCK = threading.RLock()


def _package_installed() -> bool:
    try:
        return importlib.util.find_spec("ImageReward") is not None
    except (ImportError, ValueError):
        return False


def preference_reward_status() -> dict[str, Any]:
    installed = _package_installed()
    return {
        "enabled": _ENABLED,
        "package_installed": installed,
        "available": _ENABLED and installed,
        "loaded": _MODEL is not None,
        "model": _MODEL_NAME,
        "device": _DEVICE,
        "load_error": _LOAD_ERROR,
        "message": (
            "ImageReward is enabled as an optional local human-preference signal."
            if _ENABLED and installed else
            "ImageReward is disabled. Enable BEST_IMAGE_ENABLE_PREFERENCE_REWARD=1 and install requirements-reward.txt to use it."
        ),
    }


def reward_scores_to_display(raw_scores: list[float]) -> list[float]:
    """Map ImageReward's roughly z-scaled rewards to a monotonic 0–100 display index."""
    output = []
    for score in raw_scores:
        bounded = max(-8.0, min(8.0, float(score)))
        output.append(round(100.0 / (1.0 + math.exp(-bounded)), 2))
    return output


def _load_model() -> Any:
    global _MODEL, _TORCH, _DEVICE, _LOAD_ERROR
    if _MODEL is not None:
        return _MODEL
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            import torch
            import ImageReward as reward_package

            mps = getattr(torch.backends, "mps", None)
            _DEVICE = "mps" if mps is not None and mps.is_available() else "cpu"
            _MODEL = reward_package.load(_MODEL_NAME, device=_DEVICE)
            _MODEL.eval()
            _TORCH = torch
            _LOAD_ERROR = None
            return _MODEL
        except Exception as exc:
            _LOAD_ERROR = type(exc).__name__ + ": " + str(exc)[:300]
            raise RuntimeError("ImageReward could not be loaded. Check optional dependencies, disk space, and first-download network access.") from exc


def unload_preference_reward() -> None:
    """Release optional weights before loading another large vision model."""
    global _MODEL, _TORCH, _DEVICE
    with _LOCK:
        model = _MODEL
        torch = _TORCH
        _MODEL = None
        _TORCH = None
        _DEVICE = None
        if torch is not None and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        del model
    import gc
    gc.collect()


def score_preference_reward(prompt: str, images: list[Any]) -> dict[str, Any]:
    """Score a prompt and in-memory PIL images with ImageReward when explicitly enabled."""
    status = preference_reward_status()
    if not status["enabled"]:
        return {
            "available": False, "used": False, "scores": None,
            "message": status["message"], "warning": None,
        }
    if not status["package_installed"]:
        return {
            "available": False, "used": False, "scores": None,
            "message": status["message"],
            "warning": "ImageReward is enabled but its optional Python package is missing. Install requirements-reward.txt, restart the local server, and retry.",
        }
    if len(images) < 2:
        return {
            "available": False, "used": False, "scores": None,
            "message": "ImageReward needs at least two candidate images for comparison.",
            "warning": None,
        }

    try:
        with _LOCK:
            model = _load_model()
            if _TORCH is None:
                raise RuntimeError("ImageReward's torch runtime is unavailable")
            with _TORCH.no_grad():
                _, raw = model.inference_rank(prompt, images)
            raw_scores = [float(value) for value in raw]
            if len(raw_scores) != len(images):
                raise RuntimeError("ImageReward returned a different number of scores than the candidate count")
            scores = reward_scores_to_display(raw_scores)
        return {
            "available": True,
            "used": True,
            "scores": scores,
            "raw_scores": raw_scores,
            "message": "ImageReward human-preference signal was added.",
            "warning": "ImageReward estimates learned human preference, not objective correctness. Verify exact text, object counts, and spatial details visually.",
        }
    except Exception as exc:
        unload_preference_reward()
        return {
            "available": False,
            "used": False,
            "scores": None,
            "message": "Optional ImageReward failed: " + type(exc).__name__,
            "warning": "ImageReward could not run, so the baseline ranking is retained. Check the optional install and available RAM.",
        }

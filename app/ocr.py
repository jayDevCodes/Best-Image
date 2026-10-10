"""Optional local OCR for quoted text requirements.

No network inference or image persistence is used. OCR is skipped unless the prompt
asks for image text and contains a literal phrase that can be compared reliably.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import importlib.util
import os
import re
import shutil
from typing import Any

_TEXT_TERMS = re.compile(r"\b(text|spell(?:ing)?|lettering|logo|sign|caption|written|words?|says?|reads?|reading|printed)\b", re.I)
_QUOTED = re.compile(r"""["“]([^"”\n]{2,80})["”]|'([^'\n]{2,80})'""")


def ocr_status() -> dict[str, Any]:
    configured_path = os.getenv("TESSERACT_CMD", "").strip()
    executable = configured_path if configured_path and os.path.isfile(configured_path) else shutil.which("tesseract")
    wrapper_available = importlib.util.find_spec("pytesseract") is not None
    available = bool(executable and wrapper_available)
    return {
        "available": available,
        "executable_found": bool(executable),
        "python_wrapper_installed": wrapper_available,
        "message": (
            "Local Tesseract OCR is ready."
            if available else
            "Optional OCR is unavailable. Install Tesseract and the pytesseract Python package to score quoted text."
        ),
    }


def needs_text_ocr(prompt: str) -> bool:
    return bool(_TEXT_TERMS.search(prompt or ""))


def extract_expected_text(prompt: str) -> list[str]:
    seen: set[str] = set()
    phrases: list[str] = []
    for match in _QUOTED.finditer(prompt or ""):
        phrase = (match.group(1) or match.group(2) or "").strip()
        normalized = re.sub(r"[^a-z0-9]+", "", phrase.casefold())
        if len(normalized) >= 2 and normalized not in seen:
            seen.add(normalized)
            phrases.append(phrase)
    return phrases[:5]


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").casefold())


def score_ocr_text(expected: str, recognized: str) -> float:
    """Return a comparative literal-text match signal from 0–100."""
    target = _normalize_text(expected)
    if not target:
        return 50.0
    actual = _normalize_text(recognized)
    if not actual:
        return 0.0
    if target in actual:
        return 100.0

    # OCR may return surrounding sign text. Compare against line/window fragments,
    # not only the whole page text, so unrelated text does not dominate the score.
    target_words = re.findall(r"[a-z0-9]+", expected.casefold())
    actual_words = re.findall(r"[a-z0-9]+", recognized.casefold())
    target_word_count = max(1, len(target_words))
    best = SequenceMatcher(None, target, actual).ratio()
    for line in (recognized or "").splitlines():
        line_norm = _normalize_text(line)
        if line_norm:
            best = max(best, SequenceMatcher(None, target, line_norm).ratio())
    for start in range(len(actual_words)):
        for width in range(max(1, target_word_count - 1), min(len(actual_words) - start, target_word_count + 2) + 1):
            fragment = "".join(actual_words[start:start + width])
            if fragment:
                best = max(best, SequenceMatcher(None, target, fragment).ratio())
    return round(max(0.0, min(100.0, best * 100.0)), 2)


def analyze_text_prompt(prompt: str, images: list[Any]) -> dict[str, Any]:
    """Score explicit quoted text if optional OCR is installed; fail open otherwise."""
    status = ocr_status()
    requested = needs_text_ocr(prompt)
    expected = extract_expected_text(prompt) if requested else []
    result: dict[str, Any] = {
        **status,
        "requested": requested,
        "used": False,
        "scores": None,
        "expected_count": len(expected),
        "message": status["message"],
        "warning": None,
    }
    if not requested:
        result["message"] = "OCR skipped because the prompt does not request visible text."
        return result
    if not status["available"]:
        result["warning"] = "The prompt mentions visible text, but optional local OCR is not installed; spelling and lettering need human review."
        return result
    if not expected:
        result["message"] = "OCR is available, but no exact quoted text was found in the prompt, so no literal-text score was applied."
        result["warning"] = result["message"]
        return result

    import pytesseract
    configured_path = os.getenv("TESSERACT_CMD", "").strip()
    if configured_path and os.path.isfile(configured_path):
        pytesseract.pytesseract.tesseract_cmd = configured_path

    scores: list[float] = []
    failed = 0
    for image in images:
        try:
            recognized = pytesseract.image_to_string(image, lang=os.getenv("BEST_IMAGE_OCR_LANG", "eng"), config="--psm 6", timeout=5)
            scores.append(round(sum(score_ocr_text(phrase, recognized) for phrase in expected) / len(expected), 2))
        except Exception:
            # Neutral fallback avoids treating an OCR runtime error as proof that text is absent.
            scores.append(50.0)
            failed += 1

    result["used"] = True
    result["scores"] = scores
    result["message"] = "Quoted-text OCR score was added. OCR may misread stylized or tiny text; visually verify the result."
    if failed:
        result["warning"] = "OCR failed on one or more candidates; a neutral text score was used for those images. Review them manually."
    else:
        result["warning"] = "OCR is a noisy signal and may misread stylized or tiny lettering; visually verify exact spelling."
    return result

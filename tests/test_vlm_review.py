import pytest

import app.vlm_review as vlm


def test_parser_accepts_bounded_score_and_short_reason():
    parsed = vlm.parse_review_response("SCORE: 87\nREASON: The red bicycle is visible beside the brick wall.")
    assert parsed == {"score": 87, "reason": "The red bicycle is visible beside the brick wall."}


@pytest.mark.parametrize("response", [
    "SCORE: 101\nREASON: Too high",
    "SCORE: -1\nREASON: Negative",
    "The image looks good, score is 90.",
    "SCORE: no\nREASON: Not a number",
])
def test_parser_rejects_invalid_scores(response):
    assert vlm.parse_review_response(response) is None


def test_disabled_vlm_never_invokes_model(monkeypatch):
    monkeypatch.setattr(vlm, "_ENABLED", False)
    monkeypatch.setattr(vlm, "_load_model", lambda: pytest.fail("model must remain unloaded"))
    result = vlm.score_vlm_review("A red bicycle", [object(), object()])
    assert result["used"] is False
    assert result["scores"] is None
    assert result["warning"] is None


def test_status_is_lazy_and_does_not_download_model(monkeypatch):
    monkeypatch.setattr(vlm, "_ENABLED", True)
    monkeypatch.setattr(vlm, "_dependencies_installed", lambda: True)
    monkeypatch.setattr(vlm, "_load_model", lambda: pytest.fail("status must not load weights"))
    status = vlm.vlm_review_status()
    assert status["available"] is True
    assert status["loaded"] is False
    assert "SmolVLM2" in status["model"]


def test_review_scores_preserve_candidate_order(monkeypatch):
    monkeypatch.setattr(vlm, "_ENABLED", True)
    monkeypatch.setattr(vlm, "_dependencies_installed", lambda: True)
    answers = iter([
        "SCORE: 82\nREASON: Prompt details mostly match.",
        "SCORE: 41\nREASON: Required text is absent.",
    ])
    monkeypatch.setattr(vlm, "_score_one_image", lambda prompt, image: next(answers))
    result = vlm.score_vlm_review("A sign reading OPEN", [object(), object()])
    assert result["used"] is True
    assert result["scores"] == [82.0, 41.0]
    assert result["reasons"] == ["Prompt details mostly match.", "Required text is absent."]


def test_partial_vlm_result_is_discarded_on_failure(monkeypatch):
    monkeypatch.setattr(vlm, "_ENABLED", True)
    monkeypatch.setattr(vlm, "_dependencies_installed", lambda: True)
    calls = {"count": 0}

    def fail_second(prompt, image):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated failure")
        return "SCORE: 82\nREASON: Good match."

    monkeypatch.setattr(vlm, "_score_one_image", fail_second)
    result = vlm.score_vlm_review("A red bicycle", [object(), object()])
    assert result["used"] is False
    assert result["scores"] is None
    assert result["reasons"] == []
    assert "baseline ranking was retained" in result["message"]

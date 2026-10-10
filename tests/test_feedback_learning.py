from app.feedback_store import (
    create_comparison,
    get_profile_learning,
    record_feedback,
)

BASELINE = {
    "prompt_match": 0.55,
    "requirement_match": 0.30,
    "image_clarity": 0.10,
    "technical_quality": 0.05,
}

def comparison_candidates():
    return [
        {
            "index": 0,
            "filename": "private-photo-filename-A.png",
            "criteria": {
                "prompt_match": {"score": 85},
                "requirement_match": {"score": 85},
                "image_clarity": {"score": 10},
                "technical_quality": {"score": 10},
            },
        },
        {
            "index": 1,
            "filename": "private-photo-filename-B.png",
            "criteria": {
                "prompt_match": {"score": 60},
                "requirement_match": {"score": 60},
                "image_clarity": {"score": 100},
                "technical_quality": {"score": 100},
            },
        },
    ]

def test_feedback_is_persisted_without_prompt_or_filename(monkeypatch, tmp_path):
    monkeypatch.setenv("BEST_IMAGE_DB_PATH", str(tmp_path / "feedback.sqlite3"))
    task_id = create_comparison("prompt_accuracy", "test-model", comparison_candidates())
    feedback = record_feedback(task_id, 1, "The bicycle wheels and handlebars match the prompt.")
    assert feedback["feedback_count"] == 1
    assert feedback["reason_saved"] is True
    learning = get_profile_learning("prompt_accuracy", BASELINE)
    assert learning["active"] is False
    assert learning["feedback_count"] == 1

    import sqlite3
    connection = sqlite3.connect(tmp_path / "feedback.sqlite3")
    try:
        stored, reason = connection.execute("SELECT candidates_json, feedback_reason FROM comparisons").fetchone()
    finally:
        connection.close()
    assert "private-photo-filename" not in stored
    assert '"prompt"' not in stored
    assert reason == "The bicycle wheels and handlebars match the prompt."

def test_personal_weights_activate_after_multiple_reviewed_tasks(monkeypatch, tmp_path):
    monkeypatch.setenv("BEST_IMAGE_DB_PATH", str(tmp_path / "feedback.sqlite3"))
    for _ in range(3):
        task_id = create_comparison("prompt_accuracy", "test-model", comparison_candidates())
        record_feedback(task_id, 1)

    learning = get_profile_learning("prompt_accuracy", BASELINE)
    assert learning["active"] is True
    assert learning["feedback_count"] == 3
    assert learning["informative_pairs"] >= 3
    assert learning["weights"]["image_clarity"] > BASELINE["image_clarity"]
    assert abs(sum(learning["weights"].values()) - 1.0) < 1e-5

def test_invalid_candidate_choice_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("BEST_IMAGE_DB_PATH", str(tmp_path / "feedback.sqlite3"))
    task_id = create_comparison("prompt_accuracy", "test-model", comparison_candidates())
    try:
        record_feedback(task_id, 99)
    except ValueError as exc:
        assert "does not belong" in str(exc)
    else:
        raise AssertionError("Expected invalid candidate to be rejected")

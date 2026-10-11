from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
import app.main as main_module
from app.main import app

client = TestClient(app)

def test_health_does_not_load_model_weights():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["model"]["loaded"] is False
    assert "ocr" in payload["capabilities"]
    assert "object_detection" in payload["capabilities"]
    assert "preference_reward" in payload["capabilities"]
    assert "vlm_review" in payload["capabilities"]
    assert payload["capabilities"]["vlm_review"]["used"] is False

def test_profiles_are_exposed():
    response = client.get("/api/profiles")
    assert response.status_code == 200
    profile_ids = {profile["id"] for profile in response.json()["profiles"]}
    assert {"prompt_accuracy", "visual_balance"} <= profile_ids

def test_homepage_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Find the image" in response.text

def png_bytes(color):
    output = BytesIO()
    Image.new("RGB", (64, 64), color=color).save(output, format="PNG")
    return output.getvalue()

def test_analyze_api_with_mocked_model(monkeypatch, tmp_path):
    monkeypatch.setenv("BEST_IMAGE_DB_PATH", str(tmp_path / "feedback.sqlite3"))
    def fake_score(prompt, requirements, images):
        return {
            "prompt_scores": [85.0, 68.0],
            "requirement_scores": [[82.0, 80.0], [63.0, 61.0]],
            "model": "fake-test-scorer",
            "device": "cpu",
        }
    monkeypatch.setattr(main_module.scorer, "score", fake_score)
    response = client.post(
        "/api/analyze",
        data={"prompt": "A red bicycle, beside a brick wall", "profile": "prompt_accuracy"},
        files=[
            ("images", ("candidate-a.png", png_bytes((180, 20, 30)), "image/png")),
            ("images", ("candidate-b.png", png_bytes((20, 30, 180)), "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"] == "fake-test-scorer"
    assert len(payload["ranking"]) == 2
    assert payload["ranking"][0]["filename"] == "candidate-a.png"
    assert payload["winner_index"] == 0
    assert payload["task_id"]
    assert payload["personalization"]["active"] is False
    assert "spatial relation" in " ".join(payload["warnings"])
    assert "text_match" not in payload["ranking"][0]["criteria"]
    assert "object_match" not in payload["ranking"][0]["criteria"]
    assert payload["capabilities"]["ocr"]["used"] is False
    assert payload["capabilities"]["object_detection"]["used"] is False

    feedback = client.post("/api/feedback", json={
        "task_id": payload["task_id"],
        "preferred_index": 1,
        "reason": "Candidate 2 follows the prompt more accurately; its text is spelled correctly and the exact count is three.",
    })
    assert feedback.status_code == 200, feedback.text
    assert feedback.json()["feedback_count"] == 1
    assert feedback.json()["reason_saved"] is True
    assert "prompt_match" in feedback.json()["reason_tags"]
    assert "text_or_spelling" in feedback.json()["reason_tags"]
    assert "object_counting" in feedback.json()["reason_tags"]
    assert feedback.json()["personalization"]["active"] is False

    insights = client.get("/api/feedback/insights")
    assert insights.status_code == 200, insights.text
    themes = {item["id"]: item for item in insights.json()["themes"]}
    assert themes["prompt_match"]["count"] >= 1
    assert themes["text_or_spelling"]["has_current_signal"] is False

    invalid = client.post("/api/feedback", json={
        "task_id": payload["task_id"],
        "preferred_index": 99,
    })
    assert invalid.status_code == 422


def test_cross_site_write_requests_are_blocked():
    response = client.post(
        "/api/feedback",
        json={"task_id": "nonexistent", "preferred_index": 0},
        headers={"Origin": "https://malicious.example"},
    )
    assert response.status_code == 403
    assert "Cross-site" in response.json()["detail"]


def test_extension_origin_is_allowed_through_origin_guard():
    response = client.post(
        "/api/feedback",
        json={"task_id": "nonexistent", "preferred_index": 0},
        headers={"Origin": "chrome-extension://abcdefghijklmnop"},
    )
    assert response.status_code == 404


def test_optional_visual_signals_are_added_only_when_available(monkeypatch, tmp_path):
    monkeypatch.setenv("BEST_IMAGE_DB_PATH", str(tmp_path / "feedback.sqlite3"))

    def fake_score(prompt, requirements, images):
        return {
            "prompt_scores": [80.0, 70.0],
            "requirement_scores": [[], []],
            "model": "fake-test-scorer",
            "device": "cpu",
        }

    def fake_text(prompt, images):
        return {
            "available": True, "used": True, "scores": [95.0, 10.0],
            "message": "test OCR", "warning": "test OCR warning", "expected_count": 1,
        }

    def fake_objects(prompt, images):
        return {
            "available": True, "used": True,
            "object_scores": [100.0, 0.0], "spatial_scores": [100.0, 0.0],
            "message": "test detector", "warning": "test detector warning",
        }

    monkeypatch.setattr(main_module.scorer, "score", fake_score)
    monkeypatch.setattr(main_module, "analyze_text_prompt", fake_text)
    monkeypatch.setattr(main_module, "score_object_requirements", fake_objects)
    monkeypatch.setattr(main_module, "ocr_status", lambda: {"available": True, "message": "ready"})
    monkeypatch.setattr(main_module, "detector_status", lambda: {"available": True, "message": "ready"})
    monkeypatch.setattr(main_module, "preference_reward_status", lambda: {
        "enabled": True, "package_installed": True, "available": True,
        "loaded": False, "model": "ImageReward-v1.0", "device": None, "message": "ready"
    })
    monkeypatch.setattr(main_module, "score_preference_reward", lambda prompt, images: {
        "available": True, "used": True, "scores": [88.0, 20.0],
        "message": "test preference reward", "warning": "test preference warning"
    })

    response = client.post(
        "/api/analyze",
        data={"prompt": 'A sign reading "OPEN" with exactly two bicycles to the left of a bus', "profile": "prompt_accuracy"},
        files=[
            ("images", ("candidate-a.png", png_bytes((180, 20, 30)), "image/png")),
            ("images", ("candidate-b.png", png_bytes((20, 30, 180)), "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    by_index = {item["index"]: item for item in payload["ranking"]}
    assert by_index[0]["criteria"]["text_match"]["score"] == 95.0
    assert by_index[0]["criteria"]["object_match"]["score"] == 100.0
    assert by_index[0]["criteria"]["spatial_match"]["score"] == 100.0
    assert by_index[0]["criteria"]["preference_match"]["score"] == 88.0
    assert payload["capabilities"]["preference_reward"]["used"] is True
    assert payload["capabilities"]["ocr"]["used"] is True
    assert payload["capabilities"]["object_detection"]["used"] is True
    assert any("Optional local OCR" in item for item in payload["warnings"])



def test_vlm_review_can_augment_detail_heavy_comparison(monkeypatch, tmp_path):
    monkeypatch.setenv("BEST_IMAGE_DB_PATH", str(tmp_path / "feedback.sqlite3"))

    def fake_score(prompt, requirements, images):
        return {
            "prompt_scores": [80.0, 79.0],
            "requirement_scores": [[], []],
            "model": "fake-test-scorer",
            "device": "cpu",
        }

    monkeypatch.setattr(main_module.scorer, "score", fake_score)
    monkeypatch.setattr(main_module, "vlm_review_status", lambda: {
        "enabled": True, "package_installed": True, "available": True, "loaded": False,
        "model": "test-vlm", "device": None, "load_error": None, "message": "ready",
    })
    monkeypatch.setattr(main_module, "_run_vlm_review_task", lambda prompt, images: {
        "available": True, "used": True, "scores": [40.0, 95.0],
        "reasons": ["The bus is partly occluded.", "The requested count and placement look closer."],
        "message": "test VLM", "warning": "Optional local VLM scores are heuristic.",
    })

    response = client.post(
        "/api/analyze",
        data={"prompt": 'A sign reading "OPEN" with exactly two bicycles to the left of a bus', "profile": "prompt_accuracy"},
        files=[
            ("images", ("candidate-a.png", png_bytes((180, 20, 30)), "image/png")),
            ("images", ("candidate-b.png", png_bytes((20, 30, 180)), "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    by_index = {item["index"]: item for item in payload["ranking"]}
    assert by_index[0]["criteria"]["vlm_match"]["score"] == 40.0
    assert by_index[1]["criteria"]["vlm_match"]["score"] == 95.0
    assert any("Unverified local VLM note" in note for note in by_index[1]["reasons"])
    assert payload["capabilities"]["vlm_review"]["used"] is True
    assert any("Optional local VLM scores are heuristic" in warning for warning in payload["warnings"])

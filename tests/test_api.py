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

def test_analyze_api_with_mocked_model(monkeypatch):
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
    assert "spatial relation" in " ".join(payload["warnings"])

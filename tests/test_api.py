from fastapi.testclient import TestClient
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

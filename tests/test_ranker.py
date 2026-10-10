from app.ranker import rank_candidates

def sample_candidates():
    return [
        {"index": 0, "filename": "candidate-a.png", "prompt_match": 82, "requirement_scores": [78, 60],
         "diagnostics": {"clarity_score": 90, "resolution_score": 100}},
        {"index": 1, "filename": "candidate-b.png", "prompt_match": 75, "requirement_scores": [83, 88],
         "diagnostics": {"clarity_score": 62, "resolution_score": 71}},
    ]

def test_ranker_returns_deterministic_order_and_breakdown():
    weights = {
        "prompt_match": 0.50, "requirement_match": 0.24, "object_match": 0.10,
        "spatial_match": 0.04, "text_match": 0.06, "preference_match": 0.10, "image_clarity": 0.025, "technical_quality": 0.015
    }
    first = rank_candidates(sample_candidates(), weights, ["subject", "style"])
    second = rank_candidates(sample_candidates(), weights, ["subject", "style"])
    assert [item["index"] for item in first] == [item["index"] for item in second]
    assert first[0]["rank"] == 1 and first[1]["rank"] == 2
    assert set(first[0]["criteria"]) == set(weights) - {"object_match", "spatial_match", "text_match", "preference_match"}
    assert first[0]["margin_band"] in {"low", "medium", "high"}

def test_missing_requirement_score_reweights_available_criteria():
    candidates = sample_candidates()
    for item in candidates:
        item["requirement_scores"] = []
    ranked = rank_candidates(candidates, {
        "prompt_match": 0.50, "requirement_match": 0.24,
        "object_match": 0.08, "spatial_match": 0.04, "text_match": 0.06,
        "preference_match": 0.10, "image_clarity": 0.025, "technical_quality": 0.015
    }, [])
    assert "requirement_match" not in ranked[0]["criteria"]
    assert abs(sum(value["weight"] for value in ranked[0]["criteria"].values()) - 1.0) < 1e-6

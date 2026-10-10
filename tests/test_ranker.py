from app.ranker import rank_candidates

def sample_candidates():
    return [
        {"index": 0, "filename": "candidate-a.png", "prompt_match": 82, "requirement_scores": [78, 60],
         "diagnostics": {"clarity_score": 90, "resolution_score": 100}},
        {"index": 1, "filename": "candidate-b.png", "prompt_match": 75, "requirement_scores": [83, 88],
         "diagnostics": {"clarity_score": 62, "resolution_score": 71}},
    ]

def test_ranker_returns_deterministic_order_and_breakdown():
    weights = {"prompt_match": 0.55, "requirement_match": 0.30, "image_clarity": 0.10, "technical_quality": 0.05}
    first = rank_candidates(sample_candidates(), weights, ["subject", "style"])
    second = rank_candidates(sample_candidates(), weights, ["subject", "style"])
    assert [item["index"] for item in first] == [item["index"] for item in second]
    assert first[0]["rank"] == 1 and first[1]["rank"] == 2
    assert set(first[0]["criteria"]) == set(weights)
    assert first[0]["margin_band"] in {"low", "medium", "high"}

def test_missing_requirement_score_reweights_available_criteria():
    candidates = sample_candidates()
    for item in candidates:
        item["requirement_scores"] = []
    ranked = rank_candidates(candidates, {
        "prompt_match": 0.55, "requirement_match": 0.30,
        "image_clarity": 0.10, "technical_quality": 0.05
    }, [])
    assert "requirement_match" not in ranked[0]["criteria"]
    assert abs(sum(value["weight"] for value in ranked[0]["criteria"].values()) - 1.0) < 1e-6

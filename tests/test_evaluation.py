import pytest

from app.evaluation import evaluate_rankings


def test_ranking_metrics_match_human_labels():
    report = evaluate_rankings([
        {"ranking": [{"index": 0}, {"index": 1}, {"index": 2}], "preferred_index": 1},
        {"ranking": [2, 0, 1], "preferred_index": 2},
    ])
    assert report["comparison_count"] == 2
    assert report["top1_accuracy"] == 0.5
    assert report["top3_accuracy"] == 1.0
    assert report["mean_reciprocal_rank"] == 0.75
    assert report["mean_preferred_rank"] == 1.5
    assert report["preferred_rank_histogram"] == {"1": 1, "2": 1}


@pytest.mark.parametrize("cases", [
    [],
    [{"ranking": [], "preferred_index": 0}],
    [{"ranking": [0, 0, 1], "preferred_index": 0}],
    [{"ranking": [0, 1], "preferred_index": 2}],
    [{"ranking": [0, 1], "preferred_index": "1"}],
])
def test_invalid_benchmark_cases_are_rejected(cases):
    with pytest.raises(ValueError):
        evaluate_rankings(cases)

from app.preference_reward import reward_scores_to_display


def test_preference_reward_display_score_is_monotonic_and_bounded():
    scores = reward_scores_to_display([-10.0, -1.0, 0.0, 1.0, 10.0])
    assert scores == sorted(scores)
    assert all(0.0 <= score <= 100.0 for score in scores)
    assert scores[2] == 50.0

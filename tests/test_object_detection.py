from app.object_detection import (
    _object_presence_score,
    _read_expected_count,
    _spatial_score,
    extract_count_targets,
    extract_spatial_targets,
)


def test_count_parser_ignores_candidate_number():
    prompt = "Image 2 shows several people beside a bicycle"
    person = prompt.index("people")
    assert _read_expected_count(prompt, person) is None


def test_count_targets_detect_explicit_counts_for_supported_objects():
    targets = extract_count_targets("Three red bicycles and one bus on a road")
    by_label = {item["label"]: item["expected_count"] for item in targets}
    assert by_label["bicycle"] == 3
    assert by_label["bus"] == 1


def test_simple_left_right_relation_uses_box_centers():
    relations = extract_spatial_targets("A bicycle to the left of a bus")
    assert relations
    detections = [
        {"label": "bicycle", "center_x": 20.0, "center_y": 50.0},
        {"label": "bus", "center_x": 80.0, "center_y": 50.0},
    ]
    assert _spatial_score(relations, detections) == 100.0


def test_object_count_score_penalizes_missing_and_extra_instances():
    assert _object_presence_score([{"label": "bicycle", "expected_count": 2}], [
        {"label": "bicycle"}, {"label": "bicycle"}
    ]) == 100.0
    assert _object_presence_score([{"label": "bicycle", "expected_count": 2}], [
        {"label": "bicycle"}
    ]) == 60.0

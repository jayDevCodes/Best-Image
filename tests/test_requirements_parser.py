from app.requirements_parser import extract_requirements, review_warnings

def test_extracts_simple_prompt_clauses():
    result = extract_requirements("A golden retriever on green grass, wearing a red scarf, realistic photography.")
    assert result == ["A golden retriever on green grass", "wearing a red scarf", "realistic photography"]

def test_single_clause_is_not_duplicated_as_requirement():
    assert extract_requirements("A cinematic landscape with warm light") == []

def test_count_and_text_warnings_are_explicit():
    warnings = review_warnings("Exactly 3 red signs with the word OPEN")
    assert any("counting" in item for item in warnings)
    assert any("OCR" in item for item in warnings)

def test_spatial_warning_is_available():
    assert any("spatial relation" in item for item in review_warnings("A cat to the left of a chair"))

from app.ocr import extract_expected_text, needs_text_ocr, score_ocr_text


def test_expected_literal_is_extracted_from_quotes():
    assert extract_expected_text('A storefront sign reading "OPEN 24 HOURS"') == ["OPEN 24 HOURS"]


def test_text_signal_requires_text_instruction():
    assert needs_text_ocr('A sign has the word "OPEN"')
    assert not needs_text_ocr("A red bicycle beside a brick wall")


def test_ocr_similarity_rewards_exact_literal_and_rejects_missing_text():
    assert score_ocr_text("OPEN", "Open now") == 100.0
    assert score_ocr_text("OPEN", "") == 0.0
    assert score_ocr_text("OPEN", "CLOSED") < 100.0

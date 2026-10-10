from io import BytesIO
import pytest
from PIL import Image
from app.image_processing import ImageValidationError, decode_image

def png_bytes(size=(128, 96), color=(180, 120, 40)):
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()

def test_decodes_image_and_returns_diagnostics():
    result = decode_image("photo.png", png_bytes())
    assert result.filename == "photo.png"
    assert result.width == 128 and result.height == 96
    assert 0 <= result.diagnostics["clarity_score"] <= 100
    assert 0 <= result.diagnostics["resolution_score"] <= 100

def test_rejects_empty_and_unsupported_files():
    with pytest.raises(ImageValidationError):
        decode_image("empty.png", b"")
    with pytest.raises(ImageValidationError):
        decode_image("not-an-image.svg", b"<svg></svg>")

def test_rejects_image_over_size_limit():
    with pytest.raises(ImageValidationError, match="15 MB"):
        decode_image("large.png", b"x" * (15 * 1024 * 1024 + 1))

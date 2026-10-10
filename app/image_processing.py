"""In-memory image validation and transparent technical diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import math
import warnings
from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}

class ImageValidationError(ValueError):
    """Raised when an image is too large, unsupported, or malformed."""

@dataclass
class PreparedImage:
    filename: str
    image: Image.Image
    width: int
    height: int
    byte_size: int
    diagnostics: dict[str, float | int]

def _diagnostics(image: Image.Image) -> dict[str, float | int]:
    gray = image.convert("L")
    contrast_std = float(ImageStat.Stat(gray).stddev[0])
    edge_std = float(ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).stddev[0])
    sharpness_score = min(100.0, edge_std * 2.0)
    contrast_score = min(100.0, contrast_std * 1.6)
    megapixels = (image.width * image.height) / 1_000_000
    resolution_score = min(100.0, math.sqrt((image.width * image.height) / 1_000_000) * 100.0)
    clarity_score = 0.7 * sharpness_score + 0.3 * contrast_score
    return {
        "width": image.width, "height": image.height,
        "megapixels": round(megapixels, 3),
        "sharpness_score": round(sharpness_score, 2),
        "contrast_score": round(contrast_score, 2),
        "clarity_score": round(clarity_score, 2),
        "resolution_score": round(resolution_score, 2),
    }

def decode_image(filename: str, content: bytes) -> PreparedImage:
    """Validate content by decoding it; never trust file names or MIME types."""
    if not content:
        raise ImageValidationError(f"{filename}: the uploaded file is empty")
    if len(content) > MAX_IMAGE_BYTES:
        raise ImageValidationError(f"{filename}: exceeds the 15 MB per-image limit")
    safe_name = Path(filename or "image").name[:160] or "image"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            first = Image.open(BytesIO(content))
            detected_format = first.format
            if detected_format not in ALLOWED_FORMATS:
                raise ImageValidationError(f"{safe_name}: unsupported image format; use PNG, JPEG, or WebP")
            first.verify()
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            opened = Image.open(BytesIO(content))
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ImageValidationError(f"{safe_name}: image dimensions are invalid or exceed 25 megapixels")
            corrected = ImageOps.exif_transpose(opened).convert("RGB")
            corrected.load()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError(f"{safe_name}: could not decode a safe PNG, JPEG, or WebP image") from exc
    return PreparedImage(
        filename=safe_name, image=corrected, width=corrected.width, height=corrected.height,
        byte_size=len(content), diagnostics=_diagnostics(corrected),
    )

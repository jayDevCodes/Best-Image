"""FastAPI application for the local Best-Image evaluator."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from urllib.parse import urlsplit
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_profiles
from app.feedback_store import create_comparison, get_feedback_insights, get_profile_learning, record_feedback
from app.image_processing import MAX_IMAGE_BYTES, MAX_TOTAL_UPLOAD_BYTES, ImageValidationError, PreparedImage, decode_image
from app.model import ModelUnavailableError, scorer
from app.ocr import analyze_text_prompt, ocr_status
from app.object_detection import detector_status, score_object_requirements
from app.ranker import rank_candidates
from app.requirements_parser import extract_requirements, review_warnings

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
MAX_IMAGES = 6
MIN_IMAGES = 2
MAX_PROMPT_CHARS = 2000


class FeedbackRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=64)
    preferred_index: int
    reason: str = Field(default="", max_length=2000)

app = FastAPI(
    title="Best-Image", version="0.1.0",
    description="Local prompt-to-image comparison and ranking.",
    docs_url="/api/docs", redoc_url=None,
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.middleware("http")
async def protect_local_write_endpoints(request: Request, call_next):
    """Block cross-site browser POSTs into the loopback inference service."""
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            extension_origin = parsed.scheme == "chrome-extension" and bool(parsed.netloc) and not parsed.path
            loopback_origin = (
                parsed.scheme in {"http", "https"}
                and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
                and parsed.port in {None, 8765}
            )
            if not (extension_origin or loopback_origin):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-site requests are blocked by the local Best-Image service."},
                )
    return await call_next(request)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")

@app.get("/api/health")
async def health() -> dict[str, Any]:
    # Capability checks inspect dependencies only; they do not load model weights.
    return {
        "ok": True,
        "app": "Best-Image",
        "version": "0.2.0",
        "model": scorer.status(),
        "capabilities": {
            "ocr": ocr_status(),
            "object_detection": detector_status(),
        },
    }

@app.post("/api/feedback")
async def submit_feedback(payload: FeedbackRequest) -> dict[str, Any]:
    """Record which candidate the user actually preferred for a completed task."""
    try:
        recorded = record_feedback(payload.task_id, payload.preferred_index, payload.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    configured_profiles = load_profiles()
    profile = recorded["profile"]
    learning = get_profile_learning(profile, configured_profiles[profile]["weights"])
    return {
        "ok": True,
        "message": "Your preference was saved locally. It will influence future comparisons when enough feedback is available.",
        "feedback_count": learning["feedback_count"],
        "reason_saved": recorded["reason_saved"],
        "reason_tags": recorded["reason_tags"],
        "personalization": learning,
    }


@app.get("/api/feedback/insights")
async def feedback_insights() -> dict[str, Any]:
    """Return aggregated local feedback themes, without exposing individual reasons."""
    return get_feedback_insights()


@app.get("/api/profiles")
async def profiles() -> dict[str, Any]:
    configured = load_profiles()
    return {"profiles": [
        {"id": key, "label": value["label"], "description": value["description"], "weights": value["weights"]}
        for key, value in configured.items()
    ]}

@app.post("/api/analyze")
async def analyze(
    prompt: str = Form(..., max_length=MAX_PROMPT_CHARS),
    profile: str = Form("prompt_accuracy"),
    images: list[UploadFile] = File(...),
) -> dict[str, Any]:
    cleaned_prompt = " ".join(prompt.split())
    if not cleaned_prompt:
        raise HTTPException(status_code=422, detail="Enter a prompt before analyzing images.")
    if len(images) < MIN_IMAGES or len(images) > MAX_IMAGES:
        raise HTTPException(status_code=422, detail="Upload between 2 and 6 candidate images.")

    configured_profiles = load_profiles()
    if profile not in configured_profiles:
        raise HTTPException(status_code=422, detail="Choose a valid scoring profile.")

    decoded: list[PreparedImage] = []
    total_bytes = 0
    try:
        for upload in images:
            name = upload.filename or "image"
            content = await upload.read(MAX_IMAGE_BYTES + 1)
            if len(content) > MAX_IMAGE_BYTES:
                raise ImageValidationError(f"{name}: exceeds the 15 MB per-image limit")
            total_bytes += len(content)
            if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
                raise ImageValidationError("Combined image uploads exceed the 40 MB limit")
            decoded.append(decode_image(name, content))
    except ImageValidationError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    finally:
        for upload in images:
            await upload.close()

    requirement_labels = extract_requirements(cleaned_prompt)
    base_weights = configured_profiles[profile]["weights"]
    learning = get_profile_learning(profile, base_weights)
    try:
        model_result = await run_in_threadpool(
            scorer.score, cleaned_prompt, requirement_labels, [item.image for item in decoded]
        )
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Optional checks execute locally only. Their failure never replaces the baseline scorer.
    image_list = [item.image for item in decoded]
    text_analysis = await run_in_threadpool(analyze_text_prompt, cleaned_prompt, image_list)
    object_analysis = await run_in_threadpool(score_object_requirements, cleaned_prompt, image_list)

    candidates: list[dict[str, Any]] = []
    for index, prepared in enumerate(decoded):
        candidate: dict[str, Any] = {
            "index": index,
            "filename": prepared.filename,
            "prompt_match": model_result["prompt_scores"][index],
            "requirement_scores": model_result["requirement_scores"][index],
            "diagnostics": prepared.diagnostics,
        }
        if isinstance(text_analysis.get("scores"), list):
            candidate["text_match"] = text_analysis["scores"][index]
        if isinstance(object_analysis.get("object_scores"), list):
            candidate["object_match"] = object_analysis["object_scores"][index]
        if isinstance(object_analysis.get("spatial_scores"), list):
            candidate["spatial_match"] = object_analysis["spatial_scores"][index]
        candidates.append(candidate)

    warning_list = review_warnings(
        cleaned_prompt,
        ocr_available=bool(text_analysis.get("available")),
        ocr_scored=bool(text_analysis.get("used") and text_analysis.get("scores") is not None),
        object_scored=bool(object_analysis.get("used") and object_analysis.get("object_scores") is not None),
        spatial_scored=bool(object_analysis.get("used") and object_analysis.get("spatial_scores") is not None),
    )
    for warning in (text_analysis.get("warning"), object_analysis.get("warning")):
        if isinstance(warning, str) and warning and warning not in warning_list:
            warning_list.append(warning)

    ranking = rank_candidates(candidates, learning["weights"], requirement_labels)
    task_id = create_comparison(profile, model_result["model"], ranking)
    top_gap = ranking[0]["score"] - ranking[1]["score"]
    return {
        "task_id": task_id,
        "prompt": cleaned_prompt,
        "profile": {
            "id": profile,
            "label": configured_profiles[profile]["label"],
            "weights": learning["weights"],
            "base_weights": base_weights,
        },
        "personalization": learning,
        "model": model_result["model"],
        "device": model_result["device"],
        "requirements": requirement_labels,
        "capabilities": {
            "ocr": {
                **ocr_status(),
                "used": bool(text_analysis.get("used")),
                "message": text_analysis.get("message"),
                "warning": text_analysis.get("warning"),
            },
            "object_detection": {
                **detector_status(),
                "used": bool(object_analysis.get("used")),
                "analysis_available": bool(object_analysis.get("available")),
                "message": object_analysis.get("message"),
                "warning": object_analysis.get("warning"),
            },
        },
        "warnings": warning_list,
        "ranking": ranking,
        "score_semantics": (
            "Scores combine prompt similarity, available optional local OCR/object signals, and image-statistic diagnostics. "
            "They are comparative heuristic indices, not calibrated probabilities or guaranteed correctness."
        ),
        "winner_index": ranking[0]["index"],
        "winner_gap": round(top_gap, 2),
    }

if os.getenv("BEST_IMAGE_DISABLE_DOCS", "").lower() in {"1", "true", "yes"}:
    app.openapi_url = None
    app.docs_url = None

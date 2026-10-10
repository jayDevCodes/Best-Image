"""FastAPI application for the local Best-Image evaluator."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_profiles
from app.feedback_store import create_comparison, get_profile_learning, record_feedback
from app.image_processing import MAX_IMAGE_BYTES, MAX_TOTAL_UPLOAD_BYTES, ImageValidationError, PreparedImage, decode_image
from app.model import ModelUnavailableError, scorer
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

app = FastAPI(
    title="Best-Image", version="0.1.0",
    description="Local prompt-to-image comparison and ranking.",
    docs_url="/api/docs", redoc_url=None,
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")

@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "app": "Best-Image", "version": "0.1.0", "model": scorer.status()}

@app.post("/api/feedback")
async def submit_feedback(payload: FeedbackRequest) -> dict[str, Any]:
    """Record which candidate the user actually preferred for a completed task."""
    try:
        recorded = record_feedback(payload.task_id, payload.preferred_index)
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
        "personalization": learning,
    }


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

    candidates: list[dict[str, Any]] = []
    for index, prepared in enumerate(decoded):
        candidates.append({
            "index": index,
            "filename": prepared.filename,
            "prompt_match": model_result["prompt_scores"][index],
            "requirement_scores": model_result["requirement_scores"][index],
            "diagnostics": prepared.diagnostics,
        })
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
        "warnings": review_warnings(cleaned_prompt),
        "ranking": ranking,
        "score_semantics": (
            "Scores are comparative heuristic indices from similarity and image statistics, "
            "not calibrated probabilities or guaranteed correctness."
        ),
        "winner_index": ranking[0]["index"],
        "winner_gap": round(top_gap, 2),
    }

if os.getenv("BEST_IMAGE_DISABLE_DOCS", "").lower() in {"1", "true", "yes"}:
    app.openapi_url = None
    app.docs_url = None

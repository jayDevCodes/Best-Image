"""Lazy, local-only SigLIP 2 scoring with an Apple Silicon MPS preference."""
from __future__ import annotations
import os
import threading
from typing import Any

MODEL_ID = os.getenv("BEST_IMAGE_MODEL", "google/siglip2-base-patch16-256")
MAX_TEXT_TOKENS = 64

class ModelUnavailableError(RuntimeError):
    """Raised when the local model cannot be loaded or executed."""

class SiglipScorer:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device: str | None = None
        self._load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def status(self) -> dict[str, Any]:
        return {"loaded": self.is_loaded, "model": MODEL_ID, "device": self._device, "load_error": self._load_error}

    def unload(self) -> None:
        """Release SigLIP weights before an optional larger local scorer is loaded."""
        import gc
        with self._lock:
            torch = self._torch
            loaded_model = self._model
            self._model = None
            self._processor = None
            self._torch = None
            self._device = None
            if torch is not None and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            del loaded_model
        gc.collect()

    def _load(self) -> None:
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            try:
                import torch
                from transformers import AutoModel, AutoProcessor
                requested = os.getenv("BEST_IMAGE_DEVICE", "auto").lower()
                if requested not in {"auto", "cpu", "mps"}:
                    raise ValueError("BEST_IMAGE_DEVICE must be auto, cpu, or mps")
                mps_backend = getattr(torch.backends, "mps", None)
                mps_available = bool(mps_backend is not None and mps_backend.is_available())
                if requested == "mps" and not mps_available:
                    raise RuntimeError("MPS was requested but is unavailable in this PyTorch build")
                device = "mps" if requested == "mps" or (requested == "auto" and mps_available) else "cpu"
                dtype = torch.float16 if device == "mps" else torch.float32
                processor = AutoProcessor.from_pretrained(MODEL_ID)
                model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype)
                model.eval()
                model.to(device)
                self._torch, self._processor, self._model = torch, processor, model
                self._device = device
                self._load_error = None
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise ModelUnavailableError(
                    "The local SigLIP 2 model could not be loaded. Check dependencies, free disk space, "
                    "first-download network access, and BEST_IMAGE_DEVICE settings."
                ) from exc

    def _score_one_text(self, text: str, images: list[Any]) -> list[float]:
        inputs = self._processor(
            text=[text] * len(images),
            images=images,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_TOKENS,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        if "pixel_values" in inputs:
            # Keep pixels in the model's dtype; MPS half precision otherwise can mismatch float32 inputs.
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=next(self._model.parameters()).dtype)
        with self._torch.inference_mode():
            output = self._model(**inputs)
            logits = output.logits_per_image.diagonal().float()
            # Sigmoid produces a bounded similarity signal, not a calibrated probability.
            values = self._torch.sigmoid(logits).detach().cpu().tolist()
        return [round(float(value) * 100.0, 2) for value in values]

    def score(self, prompt: str, requirements: list[str], images: list[Any]) -> dict[str, Any]:
        self._load()
        with self._lock:
            try:
                primary = self._score_one_text(prompt, images)
                per_requirement = [
                    self._score_one_text("A generated image that matches this prompt detail: " + requirement, images)
                    for requirement in requirements
                ]
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                if self._device != "mps":
                    raise ModelUnavailableError(
                        "Local model inference failed. Try CPU mode or check available memory."
                    ) from exc
                try:
                    if hasattr(self._torch, "mps") and hasattr(self._torch.mps, "empty_cache"):
                        self._torch.mps.empty_cache()
                    self._model.to(device="cpu", dtype=self._torch.float32)
                    self._device = "cpu"
                    primary = self._score_one_text(prompt, images)
                    per_requirement = [
                        self._score_one_text("A generated image that matches this prompt detail: " + requirement, images)
                        for requirement in requirements
                    ]
                    self._load_error = None
                except Exception as fallback_exc:
                    raise ModelUnavailableError(
                        "Local model inference failed on MPS and CPU fallback. Check installed model libraries and memory."
                    ) from fallback_exc
        requirement_by_image = [[scores[index] for scores in per_requirement] for index in range(len(images))]
        return {
            "prompt_scores": primary,
            "requirement_scores": requirement_by_image,
            "model": MODEL_ID,
            "device": self._device,
        }

scorer = SiglipScorer()

# Best-Image architecture

## Runtime flow

1. The local web UI or Chrome Companion supplies a prompt, scoring profile, and 2–6 candidate images. Browser images are sent to the loopback API only after user action.
2. FastAPI validates image count, format, bytes, pixels, and prompt length, then decodes images in memory.
3. The conservative prompt parser extracts up to five clauses at punctuation boundaries and flags likely count, text, and spatial requirements. It does not claim to understand every semantic constraint.
4. The required, lazily loaded SigLIP 2 scorer calculates full-prompt and clause similarities. MPS is preferred when available; an inference failure can retry on CPU.
5. Image diagnostics compute resolution adequacy, grayscale contrast, and edge-based sharpness.
6. Optional local OCR checks literal quoted text when Tesseract is available; the optional YOLOv8n detector estimates supported COCO object presence/counts and simple left/right/above/below relations; optional ImageReward adds a learned human-preference signal. These specialist models are capability-dependent and unloaded between large-model stages to reduce peak memory.
7. If enabled, the optional SmolVLM2 256M reviewer runs only when baseline scores are close or the prompt includes fragile count/text/spatial constraints. It scores each in-memory image against the prompt and contributes a separately labelled heuristic criterion. Loading, parsing, or inference failure leaves the baseline ranking intact.
8. The ranker normalizes weights over only the criteria available for that comparison. The API returns candidate order, score breakdown, inferred-detail signals, capability use, and cautionary warnings.
9. Each completed comparison receives a random task ID. Feedback stores criterion values, the chosen candidate, and optional user rationale locally; a regularized pairwise learner adapts the balance between available criteria.

## Component boundaries

- app/main.py: HTTP routes, upload validation, capability orchestration, optional review trigger, and local UI entry point.
- app/model.py: lazy SigLIP lifecycle, device choice, inference lock, and CPU fallback.
- app/vlm_review.py: optional SmolVLM2 reviewer, strict score parsing, lazy download, and graceful failure.
- app/ocr.py and app/object_detection.py: optional text and supported-object signals.
- app/preference_reward.py: optional ImageReward signal.
- app/image_processing.py: file decoding, upload safeguards, and non-persistent image handling.
- app/requirements_parser.py: conservative prompt clauses and warnings.
- app/ranker.py, app/config.py, and config/scoring_profiles.json: active-signal scoring, weight normalization, and user-editable profiles.
- app/feedback_store.py and app/feedback_insights.py: local SQLite preference learning and aggregated rationale tags.
- app/evaluation.py and scripts/evaluate_benchmark.py: offline ranking metrics for human-labelled JSONL.
- web/: dependency-free local browser UI; tests/: model-free contract, ranking, API, feedback, and optional-signal tests.

## Score semantics

Scores are bounded heuristic indices, not calibrated probabilities. SigLIP similarity and generated VLM scores can miss constraints or hallucinate; OCR can misread stylized text; the COCO detector only supports known classes and may miss small or occluded objects; ImageReward estimates learned preference rather than objective correctness. A missing signal is omitted and remaining weights are renormalized. All winners need human review where exact lettering, counts, relations, or exclusions matter.

## Feedback and personalization

The feedback store saves candidate indices and criterion score features, the selected candidate index, a random task ID, profile/model identifiers, and an optional user-entered rationale (up to 2,000 characters). It does not store prompts, filenames, or full images. A regularized Bradley–Terry-style learner activates after at least three reviewed comparisons and three informative pairwise differences, adapting criterion weights rather than fine-tuning a foundation model. Deferred reviews from the Chrome extension keep small, bounded thumbnails and metadata in chrome.storage.local until saved or dismissed; they do not rely on the old page's DOM remaining alive.

## Resource and privacy behavior

- Model weights download lazily only when a capability is used; the first download needs internet access.
- Do not enable the object detector, ImageReward, and VLM simultaneously on a memory-constrained machine just because all options exist. The pipeline runs specialists sequentially; each optional stage may add latency and can fail back to baseline.
- Images are capped at 15 MB each, 40 MB combined, 25 megapixels each, and six candidates.
- Images are processed in memory and are not saved by the application. Only explicit feedback metadata/score features are persisted outside the repository.
- The service binds to loopback by default. Do not expose it to an untrusted network.

## Validation status and remaining work

CI deliberately avoids downloading model weights. It validates API contracts, parsers, scoring behavior, storage boundaries, frontend syntax, and safety logic. CI passing does not establish real-device inference latency or human-labelled accuracy. The next meaningful evaluation is a representative held-out dataset and a real macOS/MPS resource test, followed by score calibration only if measured results justify it.

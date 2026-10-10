# Best-Image architecture (v0.1)

## Runtime flow

1. The browser UI collects a prompt, a scoring profile, and 2–6 local image files.
2. FastAPI validates image size/count/format and decodes images in memory.
3. The prompt parser creates up to five clauses at explicit punctuation boundaries. It does not claim semantic understanding.
4. The lazy SigLIP 2 scorer compares the full prompt and each inferred clause against each image. MPS is preferred when available; CPU fallback is attempted after an MPS inference failure.
5. Image diagnostics calculate edge-based sharpness, grayscale contrast, and resolution adequacy. These are interpretable statistics, not learned aesthetics.
6. The ranker combines available criteria with the selected profile. If no clause criteria exist, its weight is redistributed over remaining criteria.
7. The API returns a ranking, scores, inferred-clause scores, diagnostic notes, and review warnings.

## Component boundaries

- app/main.py: HTTP routes, request validation, orchestration, local UI.
- app/model.py: lazy model lifecycle, local device selection, inference lock, bounded CPU retry after an MPS inference failure.
- app/image_processing.py: file decoding, upload safeguards, no persistent image storage.
- app/requirements_parser.py: conservative clauses and warnings for counts, text, and spatial relationships.
- app/ranker.py: deterministic weighted scoring and explanations.
- app/config.py and config/scoring_profiles.json: profile definitions and validation.
- web/: dependency-free HTML/CSS/JavaScript frontend.
- tests/: model-free tests for contracts, ranking, image validation, and API readiness.

## Score semantics

Each score is a 0–100 heuristic index. The SigLIP score is sigmoid-transformed model similarity; it is not a calibrated probability that the image is correct. Requirement match averages scores from prompt clauses inferred by a delimiter parser. Image clarity is based on edge and contrast statistics; technical quality is based on a resolution reference. Combined rank scores should only be compared among images in the same task/profile.

Object counts, exact text, logos, fine spatial reasoning, hands/faces defects, and aesthetics are not independently verified reliably by this baseline. The API emits warnings where prompt text suggests count, lettering, or spatial requirements. Human review is recommended for those cases.

## Resource behavior

- Model download is lazy: the app server and tests do not download weights on startup.
- The base SigLIP 2 checkpoint is approximately 1.5 GB on disk; first run needs internet and additional cache space.
- Model inference is serialized so concurrent requests do not use the model at once.
- Images are capped at 15 MB each, 40 MB total, 25 megapixels per image, and six images per request.
- The default host is loopback only. Do not bind to a public interface without authentication and a threat review.

## Roadmap

- v0.2: browser extension with explicit host permissions, manual fallback for dynamic pages, and no automatic task submission.
- v0.3: opt-in OCR and optional VLM adjudication for close/complex cases; keep specialists task-scoped and measure peak memory.
- v0.4: consent-based local preference storage, labelled evaluation set, calibration, and regression benchmarks.

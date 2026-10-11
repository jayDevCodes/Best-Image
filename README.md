# Best-Image

Local-first AI image ranking for prompt-to-image comparisons.

Best-Image compares multiple generated images against a prompt and returns a ranked, explainable report. It includes a local web UI, FastAPI API, SigLIP 2 similarity scoring, image diagnostics, configurable ranking profiles, local preference learning, and opt-in local OCR and object-detection signals.

> **Important:** Best-Image is a ranking assistant, not an authoritative judge. Scores are comparative signals, not calibrated probabilities. Optional OCR can misread stylized text; the COCO detector only covers known classes, can miss small or occluded objects, and estimates only simple 2-D relations. Always inspect the suggested winner.

## Quick start (macOS / Apple Silicon)

```bash
git clone https://github.com/jayDevCodes/Best-Image.git
cd Best-Image
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q
python -m app
```

Open http://127.0.0.1:8765. The first analysis downloads the model (roughly 1.5 GB of weights plus tokenizer/configuration files); later runs reuse the Hugging Face cache. Model inference is local after the cache is populated.

If MPS inference has issues, use CPU mode:

```bash
BEST_IMAGE_DEVICE=cpu python -m app
```

## Features

- Paste a prompt and upload 2–6 candidate images (PNG, JPEG, WebP).
- Local model: google/siglip2-base-patch16-256 (Apache-2.0 per its model card).
- Compare overall prompt match and inferred prompt clauses.
- Optional quoted-text OCR scoring when Tesseract is installed and the prompt contains a literal phrase in quotes.
- Optional COCO object-presence/count and simple left/right/above/below signals when YOLOv8n is explicitly enabled.
- Transparent resolution, contrast, and edge-sharpness diagnostics.
- Custom scoring profiles in config/scoring_profiles.json.
- JSON API, health endpoint, ranking tests; no model download needed for tests.
- No remote inference API, analytics, or permanent image storage by default.
- Optional capabilities remain graceful when local dependencies are absent; detector weights are not downloaded unless you enable the detector.

## Optional text and object checks

**Local OCR (quoted text only):** On macOS with Homebrew, install Tesseract with `brew install tesseract`. The Python wrapper is installed by `requirements.txt`. OCR is scored only when the prompt requests visible text and includes a literal phrase in quotes, such as `a sign reading "OPEN"`. For a custom installation, set `TESSERACT_CMD=/path/to/tesseract`. OCR may misread stylized or small lettering, so it is an additional signal—not a guarantee.

**Object and simple spatial checks:** These are optional because they download additional weights and use extra RAM. Install them with `pip install -r requirements-vision.txt`, then start the app with `BEST_IMAGE_ENABLE_OBJECT_DETECTION=1 python -m app`. The first YOLOv8n use downloads model weights. Only supported COCO classes are detected; small or occluded objects may be missed, and depth relations such as “behind” or “in front of” are not reliably evaluated. Leave this option off if memory is limited; the baseline works without it.

The `/api/health` endpoint reports optional capability availability without loading their model weights.

**Optional human-preference reward (ImageReward):** This adds a separate preference-trained signal, but it loads a BLIP-Large backbone and can be expensive in memory and first-download time on an 8 GB Mac. Install the optional package with `pip install -r requirements-reward.txt`, then enable it for a run with `BEST_IMAGE_ENABLE_PREFERENCE_REWARD=1 python -m app`. It is disabled by default and loaded lazily. The app unloads the model after scoring to avoid keeping multiple large backbones in memory. Treat its score as a learned human-preference estimate, not proof that every requested detail is correct. If memory is tight, leave this option off (and also leave the optional detector disabled).

## Optional small vision-language review

**SmolVLM2 256M (off by default):** This is a second local reviewer for close rankings or prompts with fragile details such as exact counts, quoted text, or spatial placement. It runs only when enabled and relevant, and adds a separately labelled heuristic score; it is not proof of correctness. The model is loaded lazily and sees in-memory images and the current prompt only. The feedback database still stores criterion scores, not prompt text or image bytes.

Enable it for a run (from the project root):

    BEST_IMAGE_ENABLE_VLM_REVIEW=1 python -m app

Transformers is already part of requirements.txt. The first eligible comparison downloads HuggingFaceTB/SmolVLM2-256M-Video-Instruct into the local Hugging Face cache; subsequent inference is local. Set BEST_IMAGE_VLM_DEVICE=cpu to force CPU execution, or use auto (the default) to prefer MPS when available. This can add latency and memory use, especially on an 8 GB Mac; the implementation unloads the other local vision backbones before review and releases the VLM afterward. The score and model-generated short note may be wrong, so verify visible text, counts, and relations yourself. If loading or inference fails, the baseline ranking is retained.

## Local feedback learning

After every comparison the UI asks whether the winner was correct. You can accept it or select the candidate you actually prefer. Feedback is stored in a local SQLite database, not in the repository. It stores criterion scores, the selected candidate index, and an optional human-written feedback rationale (up to 2,000 characters), not prompts, image bytes, or filenames. After at least three informative reviewed comparisons in the same profile, a regularized pairwise learner adjusts weights for future tasks. It does not fine-tune the SigLIP foundation model.

See [docs/FEEDBACK_LEARNING.md](docs/FEEDBACK_LEARNING.md) for the learning rule, data location, reset instructions, and limitations.

## Ranking benchmark

When you have human-labelled comparisons, save them in a JSON Lines file (one JSON object per line). Each object needs the candidate order returned by the ranker and the image that was actually preferred:

```json
{"ranking":[{"index":2},{"index":0},{"index":1}],"preferred_index":0}
{"ranking":[0,1,2],"preferred_index":1}
```

Run `python scripts/evaluate_benchmark.py path/to/labels.jsonl`. The report includes top-1/top-3 accuracy, mean reciprocal rank, mean preferred rank, and a rank histogram. Keep a holdout set that is not used for tuning or preference learning; these metrics are only useful if labels are representative.

## API

- GET /api/health — readiness, SigLIP status and optional capability availability.
- GET /api/profiles — available scoring profiles.
- POST /api/analyze — multipart form with prompt, profile, and 2–6 files under images.
- POST /api/feedback — JSON with task_id, preferred_index, and optional reason (max 2,000 characters).
- GET /api/feedback/insights — aggregated local feedback topic counts, without exposing individual rationales.
- GET / — local interface.

## Scoring

Default weights are engineering defaults, not calibrated measures. Available scores are dynamically renormalized; unavailable optional signals are omitted rather than treated as zero. Requirement clauses remain similarity checks, not deterministic object detection. Every result reports criterion scores, capability use, and limitations.

## Resource and privacy notes

- Python 3.11 is recommended; PyTorch MPS is selected automatically when available.
- The model loads lazily and uses small batches. Peak memory and latency vary; close memory-heavy apps on an 8 GB Mac.
- The server binds to 127.0.0.1 by default. Image bytes are processed in memory and are not saved by the application.
- Browser-originated write requests to the API are restricted to loopback and Chrome-extension origins to reduce cross-site abuse of the local service.
- The initial model download needs internet; cached inference works locally.
- Do not expose this development server to an untrusted network.

See docs/ARCHITECTURE.md for module boundaries and roadmap.

## Project status

Implemented: local SigLIP 2 ranking, transparent image diagnostics, profile-level feedback learning, optional quoted-text OCR, optional COCO object/count and simple 2-D relation checks, optional ImageReward, optional SmolVLM2 detail review, and an offline JSONL benchmark evaluator.

Useful remaining work is evidence-driven rather than a promised feature checklist: build a representative human-labelled holdout set; measure top-1/top-3 and latency/memory on target hardware; evaluate how often optional models improve decisions; then calibrate scores only if the data supports it. CI is model-free and does not claim to benchmark real model inference on an Apple Silicon Mac.

## License

Repository code: MIT. Model weights and third-party dependencies retain their respective upstream licenses.

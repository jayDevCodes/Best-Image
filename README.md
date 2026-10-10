# Best-Image

Local-first AI image ranking for prompt-to-image comparisons.

Best-Image compares multiple generated images against a prompt and returns a ranked, explainable report. v0.1 includes a local web UI, FastAPI API, SigLIP 2 similarity scoring, image diagnostics, configurable ranking profiles, and tests.

> **Important:** v0.1 is a ranking assistant, not an authoritative judge. Similarity scores are not calibrated probabilities. Exact counts, lettering, fine spatial relations, and subtle defects can require human review.

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

## v0.1 features

- Paste a prompt and upload 2–6 candidate images (PNG, JPEG, WebP).
- Local model: google/siglip2-base-patch16-256 (Apache-2.0 per its model card).
- Compare overall prompt match and inferred prompt clauses.
- Transparent resolution, contrast, and edge-sharpness diagnostics.
- Custom scoring profiles in config/scoring_profiles.json.
- JSON API, health endpoint, ranking tests; no model download needed for tests.
- No remote inference API, analytics, or permanent image storage by default.

## Local feedback learning

After every comparison the UI asks whether the winner was correct. You can accept it or select the candidate you actually prefer. Feedback is stored in a local SQLite database, not in the repository. It stores only criterion scores and the selected candidate index, not prompts, image bytes, or filenames. After at least three informative reviewed comparisons in the same profile, a regularized pairwise learner adjusts weights for future tasks. It does not fine-tune the SigLIP foundation model.

See [docs/FEEDBACK_LEARNING.md](docs/FEEDBACK_LEARNING.md) for the learning rule, data location, reset instructions, and limitations.

## API

- GET /api/health — readiness and model status.
- GET /api/profiles — available scoring profiles.
- POST /api/analyze — multipart form with prompt, profile, and 2–6 files under images.
- GET / — local interface.

## Scoring

Default initial weights: prompt match 55%, inferred requirement match 30%, image clarity 10%, and technical quality 5%. These are engineering defaults, not calibrated measures. Requirement clauses are similarity checks, not deterministic object detection. The app exposes criterion scores and limitations with every result.

## Resource and privacy notes

- Python 3.11 is recommended; PyTorch MPS is selected automatically when available.
- The model loads lazily and uses small batches. Peak memory and latency vary; close memory-heavy apps on an 8 GB Mac.
- The server binds to 127.0.0.1 by default. Image bytes are processed in memory and are not saved by the application.
- The initial model download needs internet; cached inference works locally.
- Do not expose this development server to an untrusted network.

See docs/ARCHITECTURE.md for module boundaries and roadmap.

## Roadmap

1. v0.1: local evaluator and transparent baseline.
2. v0.2: local feedback capture and profile-level pairwise personalization (current).
3. v0.3: browser extension, optional OCR, and a small vision-language judge.
4. v0.4: labelled evaluation set, calibration, and benchmark reporting.

## License

Repository code: MIT. Model weights and third-party dependencies retain their respective upstream licenses.

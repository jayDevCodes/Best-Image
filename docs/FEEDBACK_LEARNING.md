# Local feedback and preference learning

## User experience

After an image comparison, the UI asks whether the recommended image matches the user's judgment. The user can accept the current winner or explicitly choose another candidate. The Chrome companion can also keep a lightweight, local thumbnail review pending across page navigation, then let the user choose the actual best candidate and capture a human-written rationale from the next page. Feedback applies only to that comparison and is saved locally.

## What is learned

Best-Image stores criterion score vectors for the candidates and the chosen candidate index. It does not store the prompt text, source filenames, or image bytes. A regularized Bradley-Terry-style pairwise learner fits a non-negative weight vector on a simplex for each scoring profile. For one feedback record, the chosen image is treated as preferred over each alternative in that same comparison.

The learned vector is anchored to the profile's existing weights. Personalization is enabled only after at least three reviewed comparisons and at least three informative pairwise differences. The new vector is then used for future tasks on the same profile. User feedback can be replaced if corrected through the API, and it does not update the foundation model weights. Free-text rationale tags are diagnostic only; they do not currently influence the ranker, and tags for OCR/counting/spatial checks are diagnostic; optional local signals are reported separately when available.

## Why this approach first

It is local, small, reversible, and testable without training a neural network. It optimizes relative preferences for the available scoring signals; it does not magically teach the model visual concepts the base scorer cannot detect. For example, if a user prefers accurate lettering but OCR is not installed or the prompt does not provide quoted expected text, feedback cannot reliably create that capability. Optional local OCR is now available; a COCO detector can also estimate supported object counts and simple 2-D relations when explicitly enabled. Neither expands the base model's class set or guarantees correctness.

## API

- `POST /api/analyze` returns a local `task_id`, the active profile weights, and personalization status.
- `POST /api/feedback` accepts JSON: `{"task_id":"...","preferred_index":1,"reason":"Optional explanation"}`. The reason is optional and capped at 2,000 characters.
- `GET /api/health` shows the local model status.
- `GET /api/feedback/insights` aggregates categorized rationale tags without returning individual explanations.

## Data location and privacy

The database is stored outside the repository:
- macOS: `~/Library/Application Support/Best-Image/feedback.sqlite3`
- Other platforms: `$XDG_DATA_HOME/Best-Image/feedback.sqlite3` or `~/.local/share/Best-Image/feedback.sqlite3`
- Set `BEST_IMAGE_DB_PATH` to override this path (useful for tests or a separate local profile).

The local database stores score features, a random task ID, model identifier, profile, preferred candidate index, an optional user-entered feedback reason (up to 2,000 characters), and derived topic tags. No prompt, filenames or full images are saved by the feedback store. The browser extension may separately retain small thumbnail previews locally until deferred feedback is submitted or discarded. The local SQLite database can be deleted to reset personalization.

## Limitations

- Three comparisons is a minimum activation threshold, not a guarantee that preferences are statistically stable.
- Feedback only changes the relative weighting of existing criteria. It does not retrain SigLIP or add new perception abilities.
- Scores remain comparative heuristics, not calibrated probabilities.

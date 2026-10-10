# Local feedback and preference learning

## User experience

After an image comparison, the UI asks whether the recommended image matches the user's judgment. The user can accept the current winner or explicitly choose another candidate. Feedback applies only to that comparison and is saved locally.

## What is learned

Best-Image stores criterion score vectors for the candidates and the chosen candidate index. It does not store the prompt text, source filenames, or image bytes. A regularized Bradley-Terry-style pairwise learner fits a non-negative weight vector on a simplex for each scoring profile. For one feedback record, the chosen image is treated as preferred over each alternative in that same comparison.

The learned vector is anchored to the profile's existing weights. Personalization is enabled only after at least three reviewed comparisons and at least three informative pairwise differences. The new vector is then used for future tasks on the same profile. User feedback can be replaced if corrected through the API, and it does not update the foundation model weights.

## Why this approach first

It is local, small, reversible, and testable without training a neural network. It optimizes relative preferences for the available scoring signals; it does not magically teach the model visual concepts the base scorer cannot detect. For example, if a user prefers accurate lettering but v0.1 has no OCR score, feedback cannot reliably create an OCR capability. Optional OCR/VLM/reward-model features can be added later.

## API

- `POST /api/analyze` returns a local `task_id`, the active profile weights, and personalization status.
- `POST /api/feedback` accepts JSON: `{"task_id":"...","preferred_index":1}`.
- `GET /api/health` shows the local model status.

## Data location and privacy

The database is stored outside the repository:
- macOS: `~/Library/Application Support/Best-Image/feedback.sqlite3`
- Other platforms: `$XDG_DATA_HOME/Best-Image/feedback.sqlite3` or `~/.local/share/Best-Image/feedback.sqlite3`
- Set `BEST_IMAGE_DB_PATH` to override this path (useful for tests or a separate local profile).

Only score features, a random task ID, model identifier, profile and preferred candidate index are stored. No prompt, filenames or images are saved by the feedback store. The local SQLite database can be deleted to reset personalization.

## Limitations

- Three comparisons is a minimum activation threshold, not a guarantee that preferences are statistically stable.
- Feedback only changes the relative weighting of existing criteria. It does not retrain SigLIP or add new perception abilities.
- Scores remain comparative heuristics, not calibrated probabilities.

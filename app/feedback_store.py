"""Private local preference store and a lightweight pairwise preference learner.

Only criterion scores and explicit user choices are persisted. No prompt text, filenames,
or image bytes are stored. Learned weights are regularized toward the selected profile
and activated only after several distinct comparisons have been reviewed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import sys
import uuid
from typing import Any

FEATURES = ("prompt_match", "requirement_match", "image_clarity", "technical_quality")
MIN_FEEDBACK_COMPARISONS = 3


def _db_path() -> Path:
    override = os.getenv("BEST_IMAGE_DB_PATH")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "Best-Image"
    else:
        root = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "Best-Image"
    return root / "feedback.sqlite3"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS comparisons (
            task_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            profile TEXT NOT NULL,
            model_id TEXT NOT NULL,
            candidates_json TEXT NOT NULL,
            preferred_index INTEGER,
            feedback_at TEXT
        )"""
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_comparisons_profile_feedback "
        "ON comparisons(profile, preferred_index)"
    )
    return connection


@contextmanager
def _database():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_comparison(profile: str, model_id: str, candidates: list[dict[str, Any]]) -> str:
    """Persist score features for a future user correction; never store image/prompt data."""
    compact: list[dict[str, Any]] = []
    for candidate in candidates:
        scores = candidate.get("criteria", {})
        compact.append({
            "index": int(candidate["index"]),
            "features": {
                name: float(scores[name]["score"])
                for name in FEATURES
                if name in scores and isinstance(scores[name], dict) and "score" in scores[name]
            },
        })
    if len(compact) < 2:
        raise ValueError("Feedback requires at least two candidates")
    task_id = str(uuid.uuid4())
    with _database() as connection:
        connection.execute(
            """INSERT INTO comparisons
               (task_id, created_at, profile, model_id, candidates_json)
               VALUES (?, ?, ?, ?, ?)""",
            (task_id, _utc_now(), profile, model_id, json.dumps(compact, separators=(",", ":"))),
        )
    return task_id


def record_feedback(task_id: str, preferred_index: int) -> dict[str, Any]:
    """Save or replace a user's preferred candidate for one comparison."""
    with _database() as connection:
        row = connection.execute(
            "SELECT profile, candidates_json FROM comparisons WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            raise KeyError("Comparison was not found. It may have expired or come from another installation.")
        candidates = json.loads(row["candidates_json"])
        candidate_indexes = {int(candidate["index"]) for candidate in candidates}
        if preferred_index not in candidate_indexes:
            raise ValueError("The selected candidate does not belong to this comparison")
        connection.execute(
            "UPDATE comparisons SET preferred_index = ?, feedback_at = ? WHERE task_id = ?",
            (preferred_index, _utc_now(), task_id),
        )
        count_row = connection.execute(
            "SELECT COUNT(*) AS n FROM comparisons WHERE profile = ? AND preferred_index IS NOT NULL",
            (row["profile"],),
        ).fetchone()
        return {
            "profile": row["profile"],
            "feedback_count": int(count_row["n"]),
            "preferred_index": preferred_index,
        }


def _project_simplex(values: list[float]) -> list[float]:
    """Euclidean projection onto non-negative weights summing to one."""
    if not values:
        return []
    ordered = sorted(values, reverse=True)
    running = 0.0
    rho = 0
    for index, value in enumerate(ordered, start=1):
        running += value
        theta = (running - 1.0) / index
        if value - theta > 0:
            rho = index
    if rho == 0:
        return [1.0 / len(values)] * len(values)
    theta = (sum(ordered[:rho]) - 1.0) / rho
    projected = [max(value - theta, 0.0) for value in values]
    total = sum(projected)
    return [value / total for value in projected] if total else [1.0 / len(values)] * len(values)


def _build_pairs(rows: list[sqlite3.Row]) -> list[list[float]]:
    pairs: list[list[float]] = []
    for row in rows:
        candidates = json.loads(row["candidates_json"])
        preferred_index = row["preferred_index"]
        by_index = {int(candidate["index"]): candidate for candidate in candidates}
        preferred = by_index.get(int(preferred_index)) if preferred_index is not None else None
        if preferred is None:
            continue
        for other in candidates:
            if int(other["index"]) == int(preferred_index):
                continue
            preferred_features = preferred.get("features", {})
            other_features = other.get("features", {})
            # If a criterion wasn't available on a task, give all candidates the same neutral value.
            difference = [
                (float(preferred_features.get(name, 50.0)) -
                 float(other_features.get(name, 50.0))) / 100.0
                for name in FEATURES
            ]
            if any(abs(value) > 1e-9 for value in difference):
                pairs.append(difference)
    return pairs


def _fit_weights(pairs: list[list[float]], baseline: dict[str, float]) -> dict[str, float]:
    """Regularized Bradley-Terry-style pairwise learning with a constrained weight simplex."""
    prior_values = [max(0.0, float(baseline.get(name, 0.0))) for name in FEATURES]
    prior_total = sum(prior_values)
    prior = [value / prior_total for value in prior_values] if prior_total else [0.25] * len(FEATURES)
    weights = prior[:]
    if not pairs:
        return dict(zip(FEATURES, weights))

    # Temperature controls sensitivity to score gaps; regularization prevents small-sample drift.
    temperature = 0.20
    regularization = 0.75
    learning_rate = 0.08
    for _ in range(180):
        gradient = [2.0 * regularization * (weights[i] - prior[i]) for i in range(len(FEATURES))]
        for difference in pairs:
            margin = sum(weights[i] * difference[i] for i in range(len(FEATURES))) / temperature
            if margin >= 40:
                sigmoid_negative_margin = 0.0
            elif margin <= -40:
                sigmoid_negative_margin = 1.0
            else:
                sigmoid_negative_margin = 1.0 / (1.0 + math.exp(margin))
            scale = sigmoid_negative_margin / temperature / len(pairs)
            for i in range(len(FEATURES)):
                gradient[i] -= scale * difference[i]
        weights = _project_simplex([
            weights[i] - learning_rate * gradient[i]
            for i in range(len(FEATURES))
        ])
    return {name: round(value, 6) for name, value in zip(FEATURES, weights)}


def get_profile_learning(profile: str, baseline: dict[str, float]) -> dict[str, Any]:
    """Return learned profile weights when enough reviewed comparisons exist."""
    with _database() as connection:
        rows = connection.execute(
            """SELECT candidates_json, preferred_index
               FROM comparisons
               WHERE profile = ? AND preferred_index IS NOT NULL
               ORDER BY feedback_at ASC""",
            (profile,),
        ).fetchall()
    feedback_count = len(rows)
    pairs = _build_pairs(rows)
    active = feedback_count >= MIN_FEEDBACK_COMPARISONS and len(pairs) >= MIN_FEEDBACK_COMPARISONS
    if active:
        learned = _fit_weights(pairs, baseline)
        source = "personalized"
    else:
        total = sum(max(0.0, float(value)) for value in baseline.values())
        learned = {
            key: (max(0.0, float(value)) / total if total else 1.0 / len(baseline))
            for key, value in baseline.items()
        }
        source = "default"
    return {
        "active": active,
        "source": source,
        "feedback_count": feedback_count,
        "minimum_feedback": MIN_FEEDBACK_COMPARISONS,
        "informative_pairs": len(pairs),
        "weights": learned,
        "message": (
            "Personalized ranking is active for this profile."
            if active else
            f"Saved locally. Review at least {MIN_FEEDBACK_COMPARISONS} informative comparisons on this profile to activate personalization."
        ),
    }

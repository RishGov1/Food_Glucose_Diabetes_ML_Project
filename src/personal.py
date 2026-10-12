"""Local append-only observations, versioned personal residual learning."""
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .forecast import ROOT, validate_readings

NUTRIENTS = ["carb", "sugar", "fibre", "protein", "fat"]


def connect(path=None):
    path = path or ROOT / "data/local/meals.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS meals (
        id TEXT PRIMARY KEY, profile TEXT NOT NULL, eaten_at TEXT NOT NULL,
        saved_at TEXT NOT NULL, payload TEXT NOT NULL)""")
    return db


def profile_key(name, demo=False):
    if not name.strip():
        raise ValueError("Enter a local profile name.")
    return ("demo-" if demo else "real-") + hashlib.sha256(name.strip().lower().encode()).hexdigest()[:24]


def history(profile, path=None, version=None):
    with closing(connect(path)) as db:
        rows = db.execute("SELECT id, eaten_at, payload FROM meals WHERE profile=? ORDER BY eaten_at, saved_at", (profile,)).fetchall()
    return [dict(json.loads(payload), id=mid, eaten_at=eaten) for mid, eaten, payload in rows
            if version is None or json.loads(payload)["version"] == version]


def vector(early, totals):
    return list(validate_readings(early, 4)) + [float(totals[k]) for k in NUTRIENTS]


def save_observation(profile, meal, observed, eaten_at, path=None):
    actual = validate_readings(observed, 2)
    validate_readings(meal["early"], 4)
    if not meal["foods"]:
        raise ValueError("Select at least one food to record a meal.")
    nutrients = np.array([meal["totals"][key] for key in NUTRIENTS], dtype=float)
    if not np.isfinite(nutrients).all() or (nutrients < 0).any():
        raise ValueError("Nutrients must be finite, nonnegative values.")
    if not np.isfinite(np.array(meal["base"] + meal["forecast"], dtype=float)).all():
        raise ValueError("A finite forecast is required before saving.")
    when = datetime.fromisoformat(eaten_at)
    if when.tzinfo is None:
        raise ValueError("Meal timestamp needs a timezone.")
    if when > datetime.now(timezone.utc):
        raise ValueError("Meal time cannot be in the future.")
    # Stable event key prevents double clicks and duplicate learning, even if readings are edited.
    key = hashlib.sha256(f"{profile}|{when.astimezone(timezone.utc).isoformat()}".encode()).hexdigest()
    payload = dict(meal, observed=actual.tolist())
    with closing(connect(path)) as db, db:
        changed = db.execute("INSERT OR IGNORE INTO meals VALUES (?, ?, ?, ?, ?)",
            (key, profile, when.astimezone(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat(), json.dumps(payload))).rowcount
    return bool(changed)


def correction(rows, early, totals):
    """Shrunk bias from the first observation; gated food-aware ridge after 8 meals.

    Historical predictions are captured before outcomes. No predicted labels are used.
    Refit from the immutable log on every new observation, so there is no stale model cache.
    """
    n = len(rows)
    if not n:
        return np.zeros(2), {"meals": 0, "method": "Population model; no personal outcomes yet"}
    X = np.array([vector(r["early"], r["totals"]) for r in rows])
    residual = np.array([np.asarray(r["observed"]) - np.asarray(r["base"]) for r in rows])
    bias = residual.sum(axis=0) / (n + 5)
    status = {"meals": n, "method": "Shrunk personal offset (provisional)", "version": f"personal-{n}"}
    prediction = bias
    if n >= 8:
        split = max(5, int(n * 0.75))
        candidate = make_pipeline(StandardScaler(), Ridge(alpha=50))
        candidate.fit(X[:split], residual[:split])
        baseline = residual[:split].sum(axis=0) / (split + 5)
        proposed = candidate.predict(X[split:])
        mae = float(np.mean(np.abs(proposed - residual[split:])))
        base_mae = float(np.mean(np.abs(baseline - residual[split:])))
        status.update(validation_meals=n-split, ridge_mae=mae, offset_mae=base_mae)
        # Keep the simpler correction unless the later-meal validation improves.
        if mae < base_mae * 0.95:
            candidate.fit(X, residual)
            prediction = candidate.predict([vector(early, totals)])[0]
            status["method"] = "Food-aware personal ridge (passed later-meal comparison)"
    return np.clip(prediction, -40, 40), status

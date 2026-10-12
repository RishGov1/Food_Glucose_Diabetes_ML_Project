from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytest
from src.d1namo_loader import infer_label, parse_datetime, extract_response
from src.forecast import validate_readings
from src.personal import save_observation, history, correction, profile_key


def meal():
    return {"early": [110, 130, 150, 145], "foods": [["Rice", 1]],
            "totals": dict(carb=40, sugar=2, fibre=3, protein=5, fat=2),
            "base": [140, 130], "forecast": [140, 130], "version": "v1"}


def test_date_parser_and_ambiguous_cohort():
    assert parse_datetime(pd.DataFrame({"datetime": ["2014:10:01 19:27:49"]}))[0] == pd.Timestamp("2014-10-01 19:27:49")
    assert infer_label("healthy/012_diabetes") is None


def test_forecast_landmark_has_no_future_input():
    start = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame({"timestamp": [start + pd.Timedelta(minutes=43), start + pd.Timedelta(minutes=46)], "glucose": [100, 999]})
    assert extract_response(frame, start, [45], 8)[45] == 100


def test_append_only_dedup_profile_and_version_isolation(tmp_path):
    db = tmp_path / "meals.db"
    key = profile_key("A", True)
    when = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    assert save_observation(key, meal(), [160, 170], when, db)
    assert not save_observation(key, meal(), [200, 210], when, db)
    assert history(key, db)[0]["observed"] == [160, 170]
    assert history(profile_key("A", False), db) == []
    assert history(key, db, version="v2") == []
    result, status = correction(history(key, db), meal()["early"], meal()["totals"])
    np.testing.assert_allclose(result, [20/6, 40/6])
    assert status["meals"] == 1


def test_invalid_measurements_rejected():
    for vals in ([np.nan, 90], [10, 120], [100, np.inf]):
        with pytest.raises(ValueError):
            validate_readings(vals, 2)


def test_personal_learning_uses_measured_residual_and_is_bounded():
    rows = [dict(meal(), observed=[300, 400]) for _ in range(10)]
    result, status = correction(rows, meal()["early"], meal()["totals"])
    assert np.max(np.abs(result)) <= 40
    assert status["validation_meals"] >= 2
    assert status["meals"] == 10


def test_opencl_backprop_matches_numpy_first_adam_step():
    from src.gpu_training import train_gpu
    rng = np.random.default_rng(3)
    X = rng.normal(size=(7, 4)).astype(np.float32)
    y = rng.normal(size=(7, 2)).astype(np.float32)
    seed = 42
    init = np.random.default_rng(seed)
    sizes = [4, 32, 16, 2]
    weights = [np.column_stack([init.normal(0, np.sqrt(2/ni), (no, ni)), np.zeros(no)]).astype(np.float32)
               for ni, no in zip(sizes[:-1], sizes[1:])]
    activations = [X]
    for i, w in enumerate(weights):
        z = activations[-1] @ w[:, :-1].T + w[:, -1]
        activations.append(np.maximum(z, 0) if i < 2 else z)
    deltas = [None, None, activations[-1] - y]
    for i in [1, 0]:
        deltas[i] = (deltas[i+1] @ weights[i+1][:, :-1]) * (activations[i+1] > 0)
    expected = []
    for i, w in enumerate(weights):
        grad = deltas[i].T @ np.column_stack([activations[i], np.ones(len(X))]) / len(X)
        grad[:, :-1] += .005 * w[:, :-1]
        expected.append(w - .003 * grad / (np.abs(grad) + 1e-8))
    actual, device = train_gpu(X, y, epochs=1, seed=seed)
    assert "GPU" in device
    for (w, b), expected_layer in zip(actual, expected):
        np.testing.assert_allclose(np.column_stack([w, b]), expected_layer, atol=2e-6, rtol=2e-5)

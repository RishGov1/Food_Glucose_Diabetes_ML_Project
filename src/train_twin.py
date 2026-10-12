"""CPU/GPU candidate selection with an untouched participant holdout."""
import json
import time
import platform
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .forecast import ROOT, FEATURES, TARGETS, Persistence, NeuralForecast, ResidualForecast
from .utils import save_json


def metrics(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)),
            "mae_60": float(mean_absolute_error(y[:, 0], pred[:, 0])),
            "mae_120": float(mean_absolute_error(y[:, 1], pred[:, 1])),
            "rmse": float(np.sqrt(mean_squared_error(y, pred)))}


def main():
    df = pd.read_csv(ROOT / "outputs/d1namo_meal_samples.csv")
    # Dense CGM cohort only: manual healthy measurements cannot supply this landmark protocol.
    work = df[df.subject_id.str.startswith("t1d_")].dropna(subset=FEATURES + TARGETS).copy()
    work = work[work[FEATURES + TARGETS].ge(20).all(axis=1) & work[FEATURES + TARGETS].le(600).all(axis=1)]
    X, y = work[FEATURES].to_numpy(), work[TARGETS].to_numpy()
    groups = work.subject_id.to_numpy()
    if len(set(groups)) < 5:
        raise RuntimeError("Need at least five participants with complete measured curves.")
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, groups))
    candidates = {
        "persistence": Persistence(),
        "ridge": ResidualForecast(make_pipeline(StandardScaler(), Ridge(alpha=20))),
        "random_forest": ResidualForecast(RandomForestRegressor(n_estimators=250, min_samples_leaf=5, max_depth=6, n_jobs=-1, random_state=42)),
        "extra_trees": ResidualForecast(ExtraTreesRegressor(n_estimators=250, min_samples_leaf=5, max_depth=6, n_jobs=-1, random_state=42)),
        "neural_network": NeuralForecast(),
    }
    cv = GroupKFold(n_splits=min(4, len(set(groups[train]))))
    results, oof = {}, {}
    for name, candidate in candidates.items():
        print(f"Training {name}...", flush=True)
        pred = np.zeros_like(y[train])
        for a, b in cv.split(X[train], y[train], groups[train]):
            fitted = clone(candidate).fit(X[train][a], y[train][a])
            pred[b] = fitted.predict(X[train][b])
        results[name] = metrics(y[train], pred)
        oof[name] = pred
        print(name, results[name], flush=True)
    selected = min(results, key=lambda k: results[k]["mae"])
    heldout = clone(candidates[selected]).fit(X[train], y[train]).predict(X[test])
    holdout_metrics = metrics(y[test], heldout)
    # OOF error reference is descriptive; it is not a calibrated patient confidence interval.
    error90 = np.quantile(np.abs(y[train] - oof[selected]), 0.9, axis=0).tolist()
    version = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for name, candidate in candidates.items():
        fitted = clone(candidate).fit(X, y)
        joblib.dump(fitted, ROOT / f"models/forecast_{name}.joblib")
        if name == selected:
            joblib.dump({"model": fitted, "version": version, "name": name, "error90": error90}, ROOT / "models/twin_forecast.joblib")
        if name == "neural_network":
            fitted.export_openvino(ROOT / "models/meal_neural.xml")
            gpu_name = fitted.training_device_
    report = {"version": version, "selected": selected, "n_meals": len(work), "n_subjects": len(set(groups)),
              "features": FEATURES, "targets": TARGETS, "cv": results, "holdout": holdout_metrics,
              "holdout_persistence": metrics(y[test], Persistence().predict(X[test])),
              "train_subjects": sorted(set(groups[train])), "holdout_subjects": sorted(set(groups[test])),
              "holdout_meals": len(test), "error90": error90, "gpu_training": gpu_name,
              "note": "Selection uses grouped CV on training participants only. Holdout tested once; deployed models refit on all data. Forecast is at minute 45, not before eating. Dense Type 1 CGM cohort, observational, small sample; insulin and activity unmodeled."}
    save_json(report, ROOT / "outputs/twin_metrics.json")
    print(json.dumps(report, indent=2), flush=True)
    benchmark()


def benchmark():
    import openvino as ov
    core = ov.Core()
    samples = np.array([[110, 130, 155, 150], [105, 115, 130, 128], [130, 160, 205, 215],
                        [115, 140, 175, 185], [250, 230, 200, 195], [75, 85, 90, 88]], dtype=np.float32)
    sample = samples[:1]
    neural = joblib.load(ROOT / "models/forecast_neural_network.joblib")
    expected = neural.predict(samples)
    hardware = {"cpu": platform.processor(), "available_devices": core.available_devices, "devices": {}}
    for device in core.available_devices:
        try:
            model = core.compile_model(str(ROOT / "models/meal_neural.xml"), device)
            outputs = np.concatenate([np.asarray(model([row[None, :]])[0]) for row in samples])
            start = time.perf_counter()
            for _ in range(20):
                output = np.asarray(model([sample])[0])
            hardware["devices"][device] = {"name": core.get_property(device, "FULL_DEVICE_NAME"),
                "latency_ms": (time.perf_counter() - start) * 1000 / 20,
                "test_cases": len(samples),
                "max_absolute_difference_mg_dl": float(np.max(np.abs(outputs - expected))),
                "verified": bool(np.allclose(outputs, expected, atol=2.0, rtol=0.01))}
        except Exception as exc:
            hardware["devices"][device] = {"verified": False, "error": str(exc)}
    save_json(hardware, ROOT / "outputs/hardware.json")
    print(json.dumps(hardware, indent=2), flush=True)


if __name__ == "__main__":
    main()

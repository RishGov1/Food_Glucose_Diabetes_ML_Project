"""Model selection on validation only, independent calibration and untouched test sets."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from twin.config import FEATURES, MODELS, PROCESSED, REPORTS, SEED, TARGETS
from twin.estimators import Persistence, ResidualModel


def scores(y, prediction):
    return {"mae": float(mean_absolute_error(y, prediction)),
            "mae_60": float(mean_absolute_error(y[:, 0], prediction[:, 0])),
            "mae_120": float(mean_absolute_error(y[:, 1], prediction[:, 1])),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r2": float(r2_score(y, prediction))}


def assert_split_integrity(data):
    partitions = list(data.split.unique())
    for i, a in enumerate(partitions):
        da = data[data.split == a]
        food_a = set(da.food_id) | set(da.second_food_id.dropna())
        for b in partitions[i+1:]:
            db = data[data.split == b]
            food_b = set(db.food_id) | set(db.second_food_id.dropna())
            assert not set(da.person_id) & set(db.person_id), "Participant leakage"
            assert not (food_a - {""}) & (food_b - {""}), "Food-family leakage"


def main():
    MODELS.mkdir(exist_ok=True)
    data_path = PROCESSED / "synthetic_meals.csv.gz"
    data = pd.read_csv(data_path)
    assert_split_integrity(data)
    chunks = {s: (d[FEATURES].to_numpy(), d[TARGETS].to_numpy()) for s, d in data.groupby("split")}
    x, y = chunks["train"]
    xv, yv = chunks["validation"]
    candidates = {
        "Persistence baseline": Persistence(),
        "Ridge regression": ResidualModel(make_pipeline(StandardScaler(), Ridge(alpha=30))),
        "Extra Trees": ResidualModel(ExtraTreesRegressor(n_estimators=140, min_samples_leaf=8, max_depth=22, n_jobs=4, random_state=SEED)),
        "Random Forest": ResidualModel(RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_depth=18, n_jobs=4, random_state=SEED)),
        "Histogram boosting (15 leaves)": ResidualModel(MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=240, max_leaf_nodes=15, l2_regularization=15, learning_rate=.08, early_stopping=False, random_state=SEED))),
        "Histogram boosting (31 leaves)": ResidualModel(MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=31, l2_regularization=25, learning_rate=.06, early_stopping=False, random_state=SEED))),
        "Neural network (64, 32)": ResidualModel(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64,32), alpha=.1, max_iter=100, early_stopping=True, random_state=SEED))),
    }
    table = []
    for name, estimator in candidates.items():
        started = time.perf_counter()
        with threadpool_limits(limits=4):
            estimator.fit(x, y)
            metrics = scores(yv, estimator.predict(xv))
        row = {"model": name, **metrics, "training_seconds": round(time.perf_counter()-started,2)}
        table.append(row)
        print(json.dumps(row), flush=True)
    leaderboard = pd.DataFrame(table).sort_values("mae")
    winner_name = leaderboard.iloc[0].model
    winner = clone(candidates[winner_name])
    with threadpool_limits(limits=4):
        winner.fit(np.concatenate([x,xv]), np.concatenate([y,yv]))
        xc, yc = chunks["calibration"]
        errors = np.abs(yc - winner.predict(xc))
        level = min(1, np.ceil((len(yc)+1)*.9)/len(yc))
        q = np.quantile(errors, level, axis=0, method="higher")
        xt, yt = chunks["test"]
        pred = winner.predict(xt)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    implementation = Path(__file__).read_bytes() + (Path(__file__).resolve().parents[1]/"twin/estimators.py").read_bytes()
    model_digest = hashlib.sha256(digest.encode()+implementation+sklearn.__version__.encode()).hexdigest()
    report = {"selected_model": winner_name, "selection": "lowest joint 60/120-minute MAE on food-and-person-disjoint validation",
              "version": "meal-prior-v1-" + model_digest[:12], "data_sha256": digest, "sklearn_version": sklearn.__version__,
              "features": FEATURES, "targets": TARGETS, "validation_leaderboard": leaderboard.to_dict(orient="records"),
              "test": scores(yt, pred), "test_persistence": scores(yt, Persistence().predict(xt)),
              "calibration_90_absolute_residual": q.tolist(),
              "test_simulated_interval_coverage": np.mean(np.abs(yt-pred) <= q, axis=0).tolist(),
              "clinical_accuracy": "NOT ESTABLISHED; all outcomes in this experiment are simulated.",
              "interval_note": "90% marginal simulator calibration only. App uses at least +/-35 mg/dL as an unvalidated uncertainty floor.",
              "split_integrity": "No exact canonical food-family or virtual-person overlap between partitions.",
              "semantic_overlap_limit": "Related dishes with different names can still overlap; this is not cuisine-disjoint validation."}
    artifact = {"model": winner, "report": report, "features": FEATURES, "q90": q,
                "min": data[FEATURES].min().to_dict(), "max": data[FEATURES].max().to_dict()}
    temporary = MODELS / "meal_prior.tmp"
    joblib.dump(artifact, temporary, compress=3)
    temporary.replace(MODELS / "meal_prior.joblib")
    leaderboard.to_csv(REPORTS / "model_comparison.csv", index=False)
    (REPORTS / "synthetic_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"carbs": xt[:,1], "baseline": xt[:,0], "actual_simulated_120": yt[:,1], "predicted_120": pred[:,1]}).to_csv(REPORTS / "synthetic_test_predictions.csv", index=False)
    print("Selected:", winner_name, "test MAE:", report["test"]["mae"], flush=True)


if __name__ == "__main__":
    main()

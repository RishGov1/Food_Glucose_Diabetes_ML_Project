from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, save_json, ensure_parent
from src.features import choose_response_features

RANDOM_STATE = 42


def models():
    # keep_empty_features preserves the feature width when some D1NAMO time points
    # are sparse or entirely absent in a particular cohort.
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", RandomForestClassifier(
                n_estimators=500, max_depth=5, min_samples_leaf=3,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }


def cv_score(model, X, y, groups):
    # D1NAMO is a small participant cohort. Use as many folds as the smallest
    # class has participants, capped at five, and keep each person in one fold.
    counts = pd.DataFrame({"y": y, "group": groups}).drop_duplicates("group").groupby("y")["group"].nunique()
    n_splits = min(5, int(counts.min())) if len(counts) >= 2 else 0
    if n_splits < 2:
        raise RuntimeError("Not enough distinct participants in both classes for grouped cross-validation.")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    all_true, all_pred, all_prob = [], [], []
    for train_idx, valid_idx in cv.split(X, y, groups):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = m.predict_proba(X.iloc[valid_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        all_true.extend(y.iloc[valid_idx].tolist())
        all_pred.extend(pred.tolist())
        all_prob.extend(prob.tolist())

    return {
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred)),
        "precision": float(precision_score(all_true, all_pred, zero_division=0)),
        "recall": float(recall_score(all_true, all_pred, zero_division=0)),
        "f1": float(f1_score(all_true, all_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(all_true, all_prob)),
        "validation": "StratifiedGroupKFold by participant; small-cohort estimate",
    }


def train_one_feature_set(name, features, df):
    work = df.dropna(subset=["label", "subject_id"]).copy()
    y = work["label"].astype(int)
    groups = work["subject_id"].astype(str)
    X = work[features].apply(pd.to_numeric, errors="coerce")

    results = {model_name: cv_score(model, X, y, groups) for model_name, model in models().items()}
    best_name = max(results, key=lambda k: (results[k]["roc_auc"], results[k]["balanced_accuracy"]))
    best = models()[best_name]
    best.fit(X, y)

    print(f"\n{name}")
    print("Features:", features)
    for model_name, metrics in results.items():
        print(model_name, {k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))})
    print("Selected:", best_name)

    metadata = {
        "name": name,
        "features": features,
        "selected_model": best_name,
        "cv_metrics": results[best_name],
        "n_rows": int(len(work)),
        "n_subjects": int(groups.nunique()),
        "target_note": "D1NAMO cohort label (healthy vs Type 1 diabetes), not a clinical diagnosis from one meal.",
    }
    return best, metadata


if __name__ == "__main__":
    cfg = load_config()
    data_path = Path(cfg["paths"]["samples_csv"])
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run: python -m src.build_d1namo_samples")

    df = pd.read_csv(data_path)
    response_features = choose_response_features(df)
    required = [f"g_{minute}" for minute in [0, 15, 40, 45, 60, 120]]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"D1NAMO sample table is missing required columns: {missing}")
    if df["label"].nunique() < 2:
        raise RuntimeError("Both healthy and diabetes-labelled D1NAMO participants are required.")

    # Main classifier uses the six glucose readings and derived response curve features.
    # Indian food composition is not used as a supervised feature because it is not
    # linked to the D1NAMO participants' actual meals and labels.
    response_model, response_meta = train_one_feature_set(
        "D1NAMO six-point glucose-response model", response_features, df
    )
    response_path = "models/d1namo_response_model.joblib"
    ensure_parent(response_path)
    joblib.dump(response_model, response_path)
    save_json(response_meta, "outputs/response_model_metrics.json")

    # PIMA has one OGTT glucose measurement, not the six-point meal curve. Keep a
    # separate compatible model for the external cross-dataset stress test.
    glucose_model, glucose_meta = train_one_feature_set(
        "D1NAMO 2-hour-glucose-only model", ["g_120"], df
    )
    ensure_parent(cfg["paths"]["glucose_model"])
    joblib.dump(glucose_model, cfg["paths"]["glucose_model"])
    save_json(glucose_meta, "outputs/glucose_model_metrics.json")
    print("\nSaved six-point response model and PIMA-compatible glucose-only model.")

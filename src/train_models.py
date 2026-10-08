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
from src.features import choose_common_features

RANDOM_STATE = 42

def models():
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=500,
                max_depth=5,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }

def cv_score(model, X, y, groups, n_splits=5):
    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    all_true, all_pred, all_prob = [], [], []

    for train_idx, valid_idx in cv.split(X, y, groups):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = m.predict_proba(X.iloc[valid_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)

        all_true.extend(y.iloc[valid_idx].tolist())
        all_pred.extend(pred.tolist())
        all_prob.extend(prob.tolist())

    metrics = {
        "accuracy": accuracy_score(all_true, all_pred),
        "balanced_accuracy": balanced_accuracy_score(all_true, all_pred),
        "precision": precision_score(all_true, all_pred, zero_division=0),
        "recall": recall_score(all_true, all_pred, zero_division=0),
        "f1": f1_score(all_true, all_pred, zero_division=0),
        "roc_auc": roc_auc_score(all_true, all_prob),
    }
    return metrics

def train_one_feature_set(name, features, df):
    work = df.dropna(subset=["label"]).copy()
    X = work[features]
    y = work["label"].astype(int)
    groups = work["subject_id"]

    results = {}
    for model_name, model in models().items():
        results[model_name] = cv_score(model, X, y, groups)

    best_name = max(
        results,
        key=lambda k: (results[k]["roc_auc"], results[k]["balanced_accuracy"])
    )
    best = models()[best_name]
    best.fit(X, y)

    print(f"\n{name}")
    print("Features:", features)
    print("Model comparison:")
    for mn, met in results.items():
        print(mn, {k: round(v, 4) for k, v in met.items()})
    print("Selected:", best_name)

    payload = {
        "name": name,
        "features": features,
        "selected_model": best_name,
        "cv_metrics": results[best_name],
        "n_rows": int(len(work)),
        "n_subjects": int(groups.nunique()),
    }
    return best, payload

if __name__ == "__main__":
    cfg = load_config()
    data_path = Path(cfg["paths"]["samples_csv"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run: python -m src.build_d1namo_samples"
        )

    df = pd.read_csv(data_path)
    common_features = choose_common_features(df)

    if not {"g_0", "g_120"}.issubset(common_features):
        raise RuntimeError(
            "D1NAMO does not provide enough common before/2h glucose observations "
            "for the cross-cohort classifier."
        )

    # Main D1NAMO model.
    common_model, common_meta = train_one_feature_set(
        "D1NAMO common-feature model", common_features, df
    )
    ensure_parent(cfg["paths"]["common_model"])
    joblib.dump(common_model, cfg["paths"]["common_model"])
    save_json(common_meta, "outputs/common_model_metrics.json")

    # PIMA-compatible external model: only 2-hour glucose.
    # This deliberately avoids inventing PIMA's missing time points.
    glucose_model, glucose_meta = train_one_feature_set(
        "D1NAMO 2-hour-glucose-only model", ["g_120"], df
    )
    ensure_parent(cfg["paths"]["glucose_model"])
    joblib.dump(glucose_model, cfg["paths"]["glucose_model"])
    save_json(glucose_meta, "outputs/glucose_model_metrics.json")

    print("\nSaved models.")

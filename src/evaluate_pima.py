from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, average_precision_score
)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, save_json, ensure_parent

def normalize_pima(df):
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "").replace("_", "")
        if key == "glucose":
            rename[c] = "Glucose"
        elif key == "outcome":
            rename[c] = "Outcome"
    out = df.rename(columns=rename).copy()

    if "Glucose" not in out.columns or "Outcome" not in out.columns:
        raise ValueError(
            "PIMA CSV must contain Glucose and Outcome columns."
        )
    return out

if __name__ == "__main__":
    cfg = load_config()
    pima_path = Path(cfg["paths"]["pima_csv"])
    model_path = Path(cfg["paths"]["glucose_model"])

    if not pima_path.exists():
        raise FileNotFoundError(
            f"PIMA file not found at {pima_path}. Put diabetes.csv there."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train the project first."
        )

    df = normalize_pima(pd.read_csv(pima_path))
    model = joblib.load(model_path)

    X = df[["Glucose"]].rename(columns={"Glucose": "g_120"})
    y = df["Outcome"].astype(int)

    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "average_precision": average_precision_score(y, prob),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "n_samples": int(len(df)),
        "positive_samples": int(y.sum()),
        "negative_samples": int((y == 0).sum()),
        "protocol_note": (
            "PIMA Glucose is 2-hour OGTT glucose; it is not identical to a "
            "2-hour post-meal D1NAMO CGM reading. This is external stress testing."
        ),
    }

    print("\nPIMA external evaluation")
    print(classification_report(y, pred, digits=4, zero_division=0))
    print("ROC-AUC:", round(metrics["roc_auc"], 4))
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))

    ensure_parent("outputs/pima_external_metrics.json")
    save_json(metrics, "outputs/pima_external_metrics.json")

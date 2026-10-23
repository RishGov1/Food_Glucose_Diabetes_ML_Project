"""Train Machine Learning Models on Real Clinical Dataset (PLOS ONE journal.pone.0302777).

Replaces synthetic demo models with models trained on real patient data from:
Tripathi et al. (2024), "Oral glucose tolerance test clearance in type 2 diabetes
patients who underwent remission following intense lifestyle modification:
A quasi-experimental study", PLOS ONE 19(5): e0302777.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "journal.pone.0302777.s001.xlsx"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Feature definitions
REMISSION_FEATURES = [
    "Duration_of_Diabete",
    "Heightcms",
    "Weightkgfirstoccurance",
    "Weightkglastoccurance",
    "Weight_loss_kg",
    "Weight_loss_pct",
    "BMIfirstoccurance",
    "BMIlastoccurance",
    "BMI_delta",
    "HbA1Cfirstoccurance",
    "HbA1Clastoccurance",
    "HbA1c_delta",
    "FastingBloodGlucosemgdlfirstoccurance",
    "FastingBloodGlucosemgdllastoccurance",
    "FBG_delta",
    "InsulinFastinguUmlfirstoccurance",
    "InsulinFastinguUmllastoccurance",
    "HOMA_IR_baseline",
    "HOMA_IR_endline",
    "HOMA_Beta_baseline",
    "HOMA_Beta_endline",
    "On_Insulin_yn",
    "On_Diabetes_Medicine_yn",
]

IMPACT_FEATURES = [
    "GTT_FBSL",
    "Duration_of_Diabete",
    "BMIlastoccurance",
    "Weightkglastoccurance",
    "HbA1Clastoccurance",
    "FastingBloodGlucosemgdllastoccurance",
    "HOMA_IR_endline",
    "HOMA_Beta_endline",
    "InsulinFastinguUmllastoccurance",
    "On_Insulin_yn",
    "On_Diabetes_Medicine_yn",
]


def load_and_preprocess_clinical_data(filepath: Path = DATA_PATH) -> pd.DataFrame:
    """Load and preprocess the PLOS ONE clinical dataset."""
    if not filepath.exists():
        raise FileNotFoundError(f"Clinical dataset not found at {filepath}")

    df = pd.read_excel(filepath)

    # Standardize binary indicators
    df["On_Insulin_yn"] = (
        df["On_Insulin_yn"].astype(str).str.strip().str.lower() == "yes"
    ).astype(int)
    df["On_Diabetes_Medicine_yn"] = (
        df["On_Diabetes_Medicine_yn"].astype(str).str.strip().str.lower() == "yes"
    ).astype(int)

    # Derived clinical metrics (weight reduction, glycemic delta)
    df["Weight_loss_kg"] = df["Weightkgfirstoccurance"] - df["Weightkglastoccurance"]
    df["Weight_loss_pct"] = (
        df["Weight_loss_kg"] / df["Weightkgfirstoccurance"].replace(0, np.nan)
    ) * 100.0
    df["BMI_delta"] = df["BMIfirstoccurance"] - df["BMIlastoccurance"]
    df["HbA1c_delta"] = df["HbA1Cfirstoccurance"] - df["HbA1Clastoccurance"]
    df["FBG_delta"] = (
        df["FastingBloodGlucosemgdlfirstoccurance"]
        - df["FastingBloodGlucosemgdllastoccurance"]
    )

    # Address negative HOMA-Beta values (artifact in HOMA equation when fasting glucose < 63 mg/dL)
    if "HOMA_Beta_baseline" in df.columns:
        df["HOMA_Beta_baseline"] = df["HOMA_Beta_baseline"].apply(
            lambda x: np.nan if pd.notnull(x) and x < 0 else x
        )
    if "HOMA_Beta_endline" in df.columns:
        df["HOMA_Beta_endline"] = df["HOMA_Beta_endline"].apply(
            lambda x: np.nan if pd.notnull(x) and x < 0 else x
        )

    # Target 1: OGTT Remission / Normal Glucose Clearance
    # GTT Cleared = 1 (remission achieved); IGT Cleared / Not Cleared = 0
    df["remission_label"] = (df["ReportInterpretation"] == "GTT Cleared").astype(int)

    # Target 2: Post-Challenge Excursion / High Impact
    # 2-hour post-load glucose >= 140 mg/dL (clinical threshold for postprandial hyperglycemia)
    df["food_impact_label"] = (df["GTT_PP2"] >= 140.0).astype(int)

    return df


def train_model(X: pd.DataFrame, y: pd.Series, path: Path, metrics_path: Path, names: list, title: str) -> dict:
    """Train a pipeline with median imputation, standardization, and logistic regression."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, prob)
    report = classification_report(y_test, pred, target_names=names)
    matrix = confusion_matrix(y_test, pred)

    joblib.dump(model, path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"REAL CLINICAL DATA METRICS — PLOS ONE (journal.pone.0302777)\n")
        f.write(f"Model: {title}\n")
        f.write(f"Total Cohort: {len(X)} Real T2D Patients (Train: {len(X_train)}, Test: {len(X_test)})\n\n")
        f.write(report)
        f.write(f"\nROC AUC: {auc:.4f}\n\nConfusion matrix:\n{matrix}\n")

    print(f"Saved {path} (ROC AUC: {auc:.4f})")

    return {
        "model_file": path.name,
        "title": title,
        "n_samples": len(X),
        "n_features": len(X.columns),
        "features": list(X.columns),
        "target_names": names,
        "roc_auc": round(float(auc), 4),
        "test_support": len(y_test),
    }


def main():
    print("=" * 65)
    print("Training Machine Learning Models on Real Clinical Dataset")
    print(f"Source: {DATA_PATH.name}")
    print("Study: PLOS ONE 19(5): e0302777 (Tripathi et al., 2024)")
    print("=" * 65)

    df = load_and_preprocess_clinical_data(DATA_PATH)
    print(f"Loaded {len(df)} patient records with {len(df.columns)} columns.")
    print(f"OGTT Remission Distribution: {df['remission_label'].value_counts().to_dict()}")
    print(f"Post-Challenge Excursion Distribution: {df['food_impact_label'].value_counts().to_dict()}")
    print("-" * 65)

    # 1. Remission & OGTT Clearance Model
    m1_meta = train_model(
        X=df[REMISSION_FEATURES],
        y=df["remission_label"],
        path=MODELS_DIR / "demo_model.joblib",
        metrics_path=MODELS_DIR / "demo_metrics.txt",
        names=["Impaired / Not Cleared", "GTT Cleared (Remission)"],
        title="Oral Glucose Tolerance Test (OGTT) Clearance & Remission Classifier",
    )

    # 2. Post-Challenge Excursion Risk Model
    m2_meta = train_model(
        X=df[IMPACT_FEATURES],
        y=df["food_impact_label"],
        path=MODELS_DIR / "food_impact_model.joblib",
        metrics_path=MODELS_DIR / "food_impact_metrics.txt",
        names=["Controlled Impact (<140 mg/dL)", "High Excursion / Spike (>=140 mg/dL)"],
        title="Post-Challenge Glycemic Excursion & High Impact Classifier",
    )

    # Save comprehensive metadata
    meta = {
        "dataset": {
            "file": DATA_PATH.name,
            "publication": "PLOS ONE 19(5): e0302777",
            "doi": "10.1371/journal.pone.0302777",
            "title": "Oral glucose tolerance test clearance in type 2 diabetes patients who underwent remission following lifestyle intervention",
            "cohort_size": len(df),
        },
        "models": {
            "demo_model": m1_meta,
            "food_impact_model": m2_meta,
        },
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved metadata to {MODELS_DIR / 'model_metadata.json'}")
    print("=" * 65)
    print("Model training on real clinical data complete!")


if __name__ == "__main__":
    main()

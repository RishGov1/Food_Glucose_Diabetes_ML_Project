"""Test suite for machine learning models trained on real clinical dataset."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from train_model import (
    DATA_PATH,
    MODELS_DIR,
    REMISSION_FEATURES,
    IMPACT_FEATURES,
    load_and_preprocess_clinical_data,
)


def test_clinical_dataset_exists_and_loads():
    """Verify real clinical Excel dataset exists and has expected rows and columns."""
    assert DATA_PATH.exists(), f"Missing dataset at {DATA_PATH}"
    df = load_and_preprocess_clinical_data(DATA_PATH)
    assert len(df) == 400
    assert "remission_label" in df.columns
    assert "food_impact_label" in df.columns
    assert "Weight_loss_kg" in df.columns
    assert "BMI_delta" in df.columns
    # Check no negative HOMA-Beta values remain after preprocessing
    assert (df["HOMA_Beta_baseline"].dropna() >= 0).all()


def test_saved_models_exist():
    """Verify model artifacts and metrics files are generated."""
    demo_model_path = MODELS_DIR / "demo_model.joblib"
    food_impact_model_path = MODELS_DIR / "food_impact_model.joblib"
    metadata_path = MODELS_DIR / "model_metadata.json"
    demo_metrics_path = MODELS_DIR / "demo_metrics.txt"

    assert demo_model_path.exists()
    assert food_impact_model_path.exists()
    assert metadata_path.exists()
    assert demo_metrics_path.exists()

    metrics_text = demo_metrics_path.read_text(encoding="utf-8")
    assert "REAL CLINICAL DATA METRICS" in metrics_text
    assert "PLOS ONE" in metrics_text
    assert "SYNTHETIC" not in metrics_text


def test_remission_model_inference():
    """Test loading and inference with the OGTT Remission Classifier."""
    model = joblib.load(MODELS_DIR / "demo_model.joblib")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    # Sample clinical patient profile
    sample_patient = pd.DataFrame([{
        "Duration_of_Diabete": 3.0,
        "Heightcms": 168.0,
        "Weightkgfirstoccurance": 82.0,
        "Weightkglastoccurance": 72.0,
        "Weight_loss_kg": 10.0,
        "Weight_loss_pct": 12.2,
        "BMIfirstoccurance": 29.0,
        "BMIlastoccurance": 25.5,
        "BMI_delta": 3.5,
        "HbA1Cfirstoccurance": 8.0,
        "HbA1Clastoccurance": 6.2,
        "HbA1c_delta": 1.8,
        "FastingBloodGlucosemgdlfirstoccurance": 140.0,
        "FastingBloodGlucosemgdllastoccurance": 95.0,
        "FBG_delta": 45.0,
        "InsulinFastinguUmlfirstoccurance": 12.0,
        "InsulinFastinguUmllastoccurance": 6.5,
        "HOMA_IR_baseline": 3.2,
        "HOMA_IR_endline": 1.5,
        "HOMA_Beta_baseline": 65.0,
        "HOMA_Beta_endline": 90.0,
        "On_Insulin_yn": 0,
        "On_Diabetes_Medicine_yn": 1,
    }])

    pred = model.predict(sample_patient[REMISSION_FEATURES])
    prob = model.predict_proba(sample_patient[REMISSION_FEATURES])

    assert pred[0] in [0, 1]
    assert 0.0 <= prob[0][1] <= 1.0


def test_post_challenge_excursion_model_inference():
    """Test loading and inference with the Post-Challenge Excursion Classifier."""
    model = joblib.load(MODELS_DIR / "food_impact_model.joblib")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    sample_excursion = pd.DataFrame([{
        "GTT_FBSL": 95.0,
        "Duration_of_Diabete": 4.0,
        "BMIlastoccurance": 25.0,
        "Weightkglastoccurance": 70.0,
        "HbA1Clastoccurance": 6.3,
        "FastingBloodGlucosemgdllastoccurance": 98.0,
        "HOMA_IR_endline": 1.6,
        "HOMA_Beta_endline": 85.0,
        "InsulinFastinguUmllastoccurance": 6.8,
        "On_Insulin_yn": 0,
        "On_Diabetes_Medicine_yn": 0,
    }])

    pred = model.predict(sample_excursion[IMPACT_FEATURES])
    prob = model.predict_proba(sample_excursion[IMPACT_FEATURES])

    assert pred[0] in [0, 1]
    assert 0.0 <= prob[0][1] <= 1.0


def test_model_metadata_structure():
    """Verify model metadata JSON content."""
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "dataset" in meta
    assert meta["dataset"]["file"] == "journal.pone.0302777.s001.xlsx"
    assert meta["dataset"]["cohort_size"] == 400
    assert "demo_model" in meta["models"]
    assert "food_impact_model" in meta["models"]
    assert meta["models"]["demo_model"]["roc_auc"] > 0.75

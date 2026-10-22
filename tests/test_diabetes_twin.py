"""Comprehensive test suite for Diabetes Digital Twin."""
import tempfile
from pathlib import Path
import pytest

from twin.config import DEFAULT_PATIENT, DEFAULT_TWIN_WEIGHTS
from twin.database import (
    ensure_patient_profile,
    get_hba1c_history,
    get_latest_weights,
    get_learning_history,
    get_meal_history,
    get_patient_summary_stats,
    init_db,
)
from twin.evaluator import evaluate_meal_response
from twin.online_model import adapt_twin_weights, generate_modeled_curve, predict_peak_rise
from twin.recommender import classify_food_item, get_recommendation_catalog, load_food_catalog
from twin.seed_data import seed_database


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        temp_path = Path(f.name)
    init_db(temp_path)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


def test_database_initialization(temp_db):
    """Verify database schema creation and profile seeding."""
    patient = ensure_patient_profile(DEFAULT_PATIENT, db_path=temp_db)
    assert patient["name"] == "Mahesh Chandran"
    assert patient["diabetes_type"] == "Type 2 Diabetes"

    weights = get_latest_weights(patient["id"], db_path=temp_db)
    assert "carb_weight" in weights
    assert weights["carb_weight"] == pytest.approx(DEFAULT_TWIN_WEIGHTS["carb_weight"])


def test_seed_database_and_history(temp_db):
    """Verify realistic history seeding including HbA1c trajectory and meals."""
    seed_database(db_path=temp_db, force_reseed=True)
    patient_id = DEFAULT_PATIENT["id"]

    hba1c_list = get_hba1c_history(patient_id, db_path=temp_db)
    assert len(hba1c_list) >= 4
    # Check HbA1c trend is improving (8.2 -> 6.8)
    assert hba1c_list[0]["hba1c_value"] == 8.2
    assert hba1c_list[-1]["hba1c_value"] == 6.8

    meals = get_meal_history(patient_id, limit=100, db_path=temp_db)
    assert len(meals) >= 20

    stats = get_patient_summary_stats(patient_id, db_path=temp_db)
    assert stats["total_meals"] >= 20
    assert stats["tir_percent"] > 60.0
    assert "Sweetened Instant Oats with Milk" in stats["intolerant_foods"]


def test_recommender_classification(temp_db):
    """Test food categorization into Allowed, Caution, and Not Recommended."""
    weights = DEFAULT_TWIN_WEIGHTS

    # Low carb, high fiber/protein food -> Should be ALLOWED
    good_food = {
        "food_name": "Palak Paneer",
        "unit_serving_carb_g": 6.0,
        "unit_serving_fibre_g": 4.0,
        "unit_serving_protein_g": 14.0,
        "unit_serving_fat_g": 12.0,
        "unit_serving_freesugar_g": 0.5,
        "digestion_hours": 2.0,
    }
    c_good = classify_food_item(good_food, weights, portion=1.0)
    assert c_good["tier"] == "ALLOWED"
    assert c_good["tolerance_score"] >= 75.0

    # High carb sweet food -> Should be NOT_RECOMMENDED
    bad_food = {
        "food_name": "Gulab Jamun",
        "unit_serving_carb_g": 60.0,
        "unit_serving_fibre_g": 0.5,
        "unit_serving_protein_g": 3.0,
        "unit_serving_fat_g": 15.0,
        "unit_serving_freesugar_g": 35.0,
        "digestion_hours": 0.8,
    }
    c_bad = classify_food_item(bad_food, weights, portion=1.0)
    assert c_bad["tier"] == "NOT_RECOMMENDED"
    assert c_bad["predicted_rise"] > 45.0


def test_evaluator_verdicts(temp_db):
    """Test Good, Moderate, and Bad classification of meal responses."""
    patient_id = DEFAULT_PATIENT["id"]
    seed_database(db_path=temp_db)

    # 1. Good meal response
    res_good = evaluate_meal_response(
        pre_glucose=110.0,
        readings={"glucose_30m": 125.0, "glucose_60m": 135.0, "glucose_120m": 115.0},
        food_name="Steamed Moong Salad",
        patient_id=patient_id,
        predicted_peak=132.0,
        db_path=temp_db
    )
    assert res_good["verdict"] == "GOOD"
    assert res_good["delta_glucose"] == 25.0

    # 2. Bad meal response (severe spike)
    res_bad = evaluate_meal_response(
        pre_glucose=120.0,
        readings={"glucose_30m": 175.0, "glucose_60m": 205.0, "glucose_120m": 185.0},
        food_name="White Rice Biryani",
        patient_id=patient_id,
        predicted_peak=190.0,
        db_path=temp_db
    )
    assert res_bad["verdict"] == "BAD"
    assert res_bad["delta_glucose"] == 85.0


def test_emerging_intolerance_detection(temp_db):
    """
    Test that the Digital Twin flags an emerging intolerance when a person
    previously tolerated a food with small spikes, but recently exhibits a sharp spike.
    """
    patient_id = DEFAULT_PATIENT["id"]
    seed_database(db_path=temp_db)

    # In seed data, 'Sweetened Instant Oats with Milk' was eaten multiple times:
    # 28 days ago: delta +24 mg/dL
    # 20 days ago: delta +26 mg/dL
    # If eaten today with a spike of +58 mg/dL (pre=120, peak=178):
    res = evaluate_meal_response(
        pre_glucose=120.0,
        readings={"glucose_15m": 140.0, "glucose_30m": 160.0, "glucose_45m": 178.0, "glucose_60m": 172.0, "glucose_120m": 154.0},
        food_name="Sweetened Instant Oats with Milk",
        patient_id=patient_id,
        predicted_peak=148.0,
        db_path=temp_db
    )

    assert res["is_emerging_intolerant"] is True
    assert res["intolerance_status"] == "EMERGING_INTOLERANT"
    assert "Developing Intolerance Detected" in res["intolerance_message"] or "Emerging Intolerance" in res["intolerance_title"]


def test_online_weight_adaptation(temp_db):
    """Test online parameter updates when observed peak exceeds prediction."""
    patient_id = DEFAULT_PATIENT["id"]
    seed_database(db_path=temp_db)

    initial_weights = get_latest_weights(patient_id, db_path=temp_db)
    initial_carb_w = initial_weights["carb_weight"]

    meal = {
        "id": 999,
        "food_name": "High Carb Test Meal",
        "carbs_g": 60.0,
        "fiber_g": 2.0,
        "protein_g": 10.0,
        "fat_g": 8.0,
        "pre_glucose": 115.0,
        "intolerance_status": "EMERGING_INTOLERANT",
    }
    # Observed peak 185 mg/dL is higher than predicted
    update_res = adapt_twin_weights(
        patient_id=patient_id,
        meal_record=meal,
        observed_peak=185.0,
        hba1c=6.8,
        current_weights=initial_weights,
        learning_rate=0.04,
        db_path=temp_db
    )

    # Weights should adapt upwards for carb sensitivity
    assert update_res["new_weights"]["carb_weight"] > initial_carb_w
    assert update_res["learning_step"] == initial_weights["learning_step"] + 1

    # Verify persisted in database
    latest_in_db = get_latest_weights(patient_id, db_path=temp_db)
    assert latest_in_db["carb_weight"] == update_res["new_weights"]["carb_weight"]

    # Verify telemetry event was recorded
    history = get_learning_history(patient_id, db_path=temp_db)
    assert len(history) >= 1
    assert history[0]["loss"] > 0.0


def test_modeled_curve_generation():
    """Verify 7-point kinetic glucose curve simulation."""
    curve = generate_modeled_curve(
        pre_glucose=115.0,
        predicted_peak_delta=35.0,
        digestion_hours=1.5,
        fat_g=6.0,
        fiber_g=4.0
    )
    assert len(curve) == 7
    times = [pt[0] for pt in curve]
    assert times == [0, 15, 30, 45, 60, 90, 120]
    # Curve starts at baseline
    assert curve[0][1] == 115.0
    # Peak occurs mid-interval and is higher than baseline
    peak_val = max(pt[1] for pt in curve)
    assert peak_val > 115.0

"""Seed initial realistic data for a diabetic patient digital twin."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from twin.config import DB_PATH, DEFAULT_PATIENT, DEFAULT_TWIN_WEIGHTS
from twin.database import (
    add_hba1c_log,
    add_meal_log,
    ensure_patient_profile,
    get_meal_history,
    init_db,
    save_weights,
)


def seed_database(db_path: Optional[Path] = None, force_reseed: bool = False) -> None:
    """Populate the database with realistic longitudinal diabetic history if empty or forced."""
    init_db(db_path)
    ensure_patient_profile(DEFAULT_PATIENT, db_path)

    patient_id = DEFAULT_PATIENT["id"]
    existing = get_meal_history(patient_id, limit=5, db_path=db_path)
    if existing and not force_reseed:
        return  # Already seeded

    # Seed Longitudinal HbA1c Progression
    hba1c_records = [
        ("2026-01-15", 8.2, 164.0, "Baseline medical checkup; newly diagnosed Type 2 diabetes."),
        ("2026-04-18", 7.6, 142.0, "Initiated 30-min brisk morning walk and portion moderation."),
        ("2026-07-20", 7.1, 128.0, "Reduced evening high-glycemic carbohydrates; notable HbA1c drop."),
        ("2026-10-10", 6.8, 118.0, "Optimal glycemic management; digital twin active guidance."),
    ]
    for date_str, val, fasting, notes in hba1c_records:
        add_hba1c_log(patient_id, date_str, val, fasting, notes, db_path)

    # Base reference date: October 22, 2026
    now = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)

    # 30+ Realistic meals across past 30 days
    historical_meals = [
        # --- Week 4 ago (Days -28 to -22) ---
        {
            "days_ago": 28, "hour": 8, "meal_type": "Breakfast",
            "food_name": "Sweetened Instant Oats with Milk", "portion": 1.0, "unit": "bowl",
            "carbs": 38.0, "fiber": 4.0, "protein": 9.0, "fat": 5.0, "digestion": 1.2,
            "pre": 114.0, "g15": 124.0, "g30": 132.0, "g45": 138.0, "g60": 135.0, "g120": 118.0,
            "peak": 138.0, "delta": 24.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Tolerated well with mild rise.", "pred": 136.0, "err": 2.0
        },
        {
            "days_ago": 28, "hour": 13, "meal_type": "Lunch",
            "food_name": "Palak Paneer with 2 Multigrain Rotis", "portion": 1.0, "unit": "plate",
            "carbs": 32.0, "fiber": 6.5, "protein": 18.0, "fat": 14.0, "digestion": 2.2,
            "pre": 116.0, "g15": 122.0, "g30": 134.0, "g45": 141.0, "g60": 138.0, "g120": 119.0,
            "peak": 141.0, "delta": 25.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Excellent fiber and protein stabilization.", "pred": 140.0, "err": 1.0
        },
        {
            "days_ago": 27, "hour": 20, "meal_type": "Dinner",
            "food_name": "Moong Dal Khichdi with Curd", "portion": 1.0, "unit": "bowl",
            "carbs": 36.0, "fiber": 5.0, "protein": 12.0, "fat": 6.0, "digestion": 1.8,
            "pre": 118.0, "g15": 128.0, "g30": 142.0, "g45": 146.0, "g60": 143.0, "g120": 122.0,
            "peak": 146.0, "delta": 28.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Gentle glycemic excursion; steady recovery.", "pred": 144.0, "err": 2.0
        },
        {
            "days_ago": 26, "hour": 13, "meal_type": "Lunch",
            "food_name": "White Rice Biryani with Raita", "portion": 1.5, "unit": "serving",
            "carbs": 84.0, "fiber": 2.2, "protein": 16.0, "fat": 22.0, "digestion": 2.5,
            "pre": 120.0, "g15": 145.0, "g30": 172.0, "g45": 194.0, "g60": 202.0, "g120": 178.0,
            "peak": 202.0, "delta": 82.0, "verdict": "BAD", "intolerance": "KNOWN_INTOLERANT",
            "notes": "Severe postprandial spike from high glycemic load.", "pred": 195.0, "err": 7.0
        },
        {
            "days_ago": 25, "hour": 8, "meal_type": "Breakfast",
            "food_name": "Moong Dal Cheela with Mint Chutney", "portion": 2.0, "unit": "piece",
            "carbs": 24.0, "fiber": 5.5, "protein": 14.0, "fat": 4.5, "digestion": 1.5,
            "pre": 112.0, "g15": 118.0, "g30": 128.0, "g45": 133.0, "g60": 129.0, "g120": 115.0,
            "peak": 133.0, "delta": 21.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Minimal spike, high protein buffering.", "pred": 131.0, "err": 2.0
        },
        {
            "days_ago": 24, "hour": 17, "meal_type": "Snack",
            "food_name": "Roasted Makhana (Fox Nuts)", "portion": 1.0, "unit": "bowl",
            "carbs": 18.0, "fiber": 3.0, "protein": 4.0, "fat": 2.0, "digestion": 1.0,
            "pre": 115.0, "g15": 121.0, "g30": 128.0, "g45": 127.0, "g60": 124.0, "g120": 116.0,
            "peak": 128.0, "delta": 13.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Very safe, low glycemic snack.", "pred": 127.0, "err": 1.0
        },

        # --- Week 3 ago (Days -21 to -15) ---
        {
            "days_ago": 20, "hour": 8, "meal_type": "Breakfast",
            "food_name": "Sweetened Instant Oats with Milk", "portion": 1.0, "unit": "bowl",
            "carbs": 38.0, "fiber": 4.0, "protein": 9.0, "fat": 5.0, "digestion": 1.2,
            "pre": 116.0, "g15": 127.0, "g30": 136.0, "g45": 142.0, "g60": 139.0, "g120": 121.0,
            "peak": 142.0, "delta": 26.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Repeated meal: consistent mild rise (~26 mg/dL).", "pred": 138.0, "err": 4.0
        },
        {
            "days_ago": 19, "hour": 13, "meal_type": "Lunch",
            "food_name": "Brown Rice with Rajma and Salad", "portion": 1.0, "unit": "plate",
            "carbs": 48.0, "fiber": 8.0, "protein": 15.0, "fat": 6.0, "digestion": 2.5,
            "pre": 117.0, "g15": 126.0, "g30": 144.0, "g45": 156.0, "g60": 152.0, "g120": 132.0,
            "peak": 156.0, "delta": 39.0, "verdict": "MODERATE", "intolerance": "TOLERATED",
            "notes": "Moderate excursion; soluble fiber kept 2-hr within limits.", "pred": 154.0, "err": 2.0
        },
        {
            "days_ago": 18, "hour": 20, "meal_type": "Dinner",
            "food_name": "Methi Thepla with Low Fat Curd", "portion": 2.0, "unit": "piece",
            "carbs": 30.0, "fiber": 5.0, "protein": 11.0, "fat": 7.0, "digestion": 1.6,
            "pre": 114.0, "g15": 122.0, "g30": 135.0, "g45": 139.0, "g60": 136.0, "g120": 118.0,
            "peak": 139.0, "delta": 25.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Methi fenugreek properties help glycemic response.", "pred": 137.0, "err": 2.0
        },
        {
            "days_ago": 17, "hour": 16, "meal_type": "Snack",
            "food_name": "Gulab Jamun (2 pieces)", "portion": 2.0, "unit": "piece",
            "carbs": 58.0, "fiber": 0.5, "protein": 4.0, "fat": 15.0, "digestion": 1.0,
            "pre": 119.0, "g15": 148.0, "g30": 182.0, "g45": 198.0, "g60": 205.0, "g120": 184.0,
            "peak": 205.0, "delta": 86.0, "verdict": "BAD", "intolerance": "KNOWN_INTOLERANT",
            "notes": "Severe sharp spike due to refined sugar syrup.", "pred": 196.0, "err": 9.0
        },
        {
            "days_ago": 16, "hour": 13, "meal_type": "Lunch",
            "food_name": "Sprouted Moong Salad with Lemon", "portion": 1.0, "unit": "bowl",
            "carbs": 22.0, "fiber": 7.0, "protein": 13.0, "fat": 2.5, "digestion": 1.4,
            "pre": 113.0, "g15": 117.0, "g30": 124.0, "g45": 129.0, "g60": 126.0, "g120": 114.0,
            "peak": 129.0, "delta": 16.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Ideal low-GI lunch option.", "pred": 128.0, "err": 1.0
        },

        # --- Week 2 ago (Days -14 to -8) ---
        {
            "days_ago": 12, "hour": 8, "meal_type": "Breakfast",
            "food_name": "Sweetened Instant Oats with Milk", "portion": 1.0, "unit": "bowl",
            "carbs": 38.0, "fiber": 4.0, "protein": 9.0, "fat": 5.0, "digestion": 1.2,
            "pre": 120.0, "g15": 134.0, "g30": 145.0, "g45": 156.0, "g60": 153.0, "g120": 134.0,
            "peak": 156.0, "delta": 36.0, "verdict": "MODERATE", "intolerance": "TOLERATED",
            "notes": "Noticeable higher rise than 2 weeks prior (+36 vs +24 mg/dL).", "pred": 142.0, "err": 14.0
        },
        {
            "days_ago": 11, "hour": 13, "meal_type": "Lunch",
            "food_name": "Paneer Bhurji with 2 Roti and Cucumber", "portion": 1.0, "unit": "plate",
            "carbs": 34.0, "fiber": 5.0, "protein": 20.0, "fat": 16.0, "digestion": 2.0,
            "pre": 116.0, "g15": 124.0, "g30": 138.0, "g45": 144.0, "g60": 142.0, "g120": 122.0,
            "peak": 144.0, "delta": 28.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Stable response; good satiety.", "pred": 143.0, "err": 1.0
        },
        {
            "days_ago": 10, "hour": 20, "meal_type": "Dinner",
            "food_name": "Yellow Moong Dal with Steamed Rice (Small portion)", "portion": 1.0, "unit": "plate",
            "carbs": 44.0, "fiber": 4.5, "protein": 11.0, "fat": 4.0, "digestion": 1.6,
            "pre": 118.0, "g15": 132.0, "g30": 152.0, "g45": 162.0, "g60": 158.0, "g120": 135.0,
            "peak": 162.0, "delta": 44.0, "verdict": "MODERATE", "intolerance": "TOLERATED",
            "notes": "Borderline 44 mg/dL rise due to rice portion.", "pred": 158.0, "err": 4.0
        },
        {
            "days_ago": 9, "hour": 8, "meal_type": "Breakfast",
            "food_name": "Idli Sambar with Coconut Chutney (3 Idlis)", "portion": 1.0, "unit": "plate",
            "carbs": 52.0, "fiber": 4.0, "protein": 8.0, "fat": 6.0, "digestion": 1.5,
            "pre": 117.0, "g15": 134.0, "g30": 156.0, "g45": 168.0, "g60": 162.0, "g120": 138.0,
            "peak": 168.0, "delta": 51.0, "verdict": "MODERATE", "intolerance": "TOLERATED",
            "notes": "Moderate excursion; portion adjustment suggested.", "pred": 163.0, "err": 5.0
        },
        {
            "days_ago": 8, "hour": 17, "meal_type": "Snack",
            "food_name": "Roasted Chana with Green Tea", "portion": 1.0, "unit": "bowl",
            "carbs": 20.0, "fiber": 5.0, "protein": 7.0, "fat": 3.0, "digestion": 1.2,
            "pre": 114.0, "g15": 119.0, "g30": 126.0, "g45": 128.0, "g60": 125.0, "g120": 116.0,
            "peak": 128.0, "delta": 14.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Very favorable glycemic stability.", "pred": 127.0, "err": 1.0
        },

        # --- Recent Week (Days -7 to -1) ---
        {
            "days_ago": 6, "hour": 13, "meal_type": "Lunch",
            "food_name": "Grilled Chicken Salad with Olive Oil", "portion": 1.0, "unit": "plate",
            "carbs": 12.0, "fiber": 4.0, "protein": 32.0, "fat": 12.0, "digestion": 2.0,
            "pre": 112.0, "g15": 116.0, "g30": 123.0, "g45": 127.0, "g60": 125.0, "g120": 114.0,
            "peak": 127.0, "delta": 15.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Virtually flat curve; exceptional control.", "pred": 125.0, "err": 2.0
        },
        {
            "days_ago": 5, "hour": 20, "meal_type": "Dinner",
            "food_name": "Roti with Dal Tadka and Bhindi Sabzi", "portion": 1.0, "unit": "plate",
            "carbs": 38.0, "fiber": 6.5, "protein": 12.0, "fat": 8.0, "digestion": 1.8,
            "pre": 115.0, "g15": 125.0, "g30": 138.0, "g45": 145.0, "g60": 141.0, "g120": 120.0,
            "peak": 145.0, "delta": 30.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Well controlled dinner response.", "pred": 144.0, "err": 1.0
        },
        # KEY EMERGING INTOLERANCE OBSERVATION:
        {
            "days_ago": 4, "hour": 8, "meal_type": "Breakfast",
            "food_name": "Sweetened Instant Oats with Milk", "portion": 1.0, "unit": "bowl",
            "carbs": 38.0, "fiber": 4.0, "protein": 9.0, "fat": 5.0, "digestion": 1.2,
            "pre": 122.0, "g15": 142.0, "g30": 158.0, "g45": 174.0, "g60": 170.0, "g120": 152.0,
            "peak": 174.0, "delta": 52.0, "verdict": "BAD", "intolerance": "EMERGING_INTOLERANT",
            "notes": "Clear developing intolerance! Spike of +52 mg/dL vs historical baseline of +24 mg/dL. Downward tolerance trend confirmed.",
            "pred": 148.0, "err": 26.0
        },
        {
            "days_ago": 3, "hour": 13, "meal_type": "Lunch",
            "food_name": "Chana Masala with 2 Phulka Rotis", "portion": 1.0, "unit": "plate",
            "carbs": 42.0, "fiber": 8.5, "protein": 14.0, "fat": 7.0, "digestion": 2.2,
            "pre": 118.0, "g15": 126.0, "g30": 140.0, "g45": 147.0, "g60": 144.0, "g120": 124.0,
            "peak": 147.0, "delta": 29.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Chickpea high fiber provides steady glycemic curve.", "pred": 145.0, "err": 2.0
        },
        {
            "days_ago": 2, "hour": 20, "meal_type": "Dinner",
            "food_name": "Lauki (Bottle Gourd) Curry with 2 Phulka and Curd", "portion": 1.0, "unit": "plate",
            "carbs": 28.0, "fiber": 5.5, "protein": 9.0, "fat": 5.0, "digestion": 1.5,
            "pre": 116.0, "g15": 122.0, "g30": 132.0, "g45": 137.0, "g60": 134.0, "g120": 118.0,
            "peak": 137.0, "delta": 21.0, "verdict": "GOOD", "intolerance": "TOLERATED",
            "notes": "Low calorie, high hydration dinner meal.", "pred": 135.0, "err": 2.0
        },
        {
            "days_ago": 1, "hour": 8, "meal_type": "Breakfast",
            "food_name": "Vegetable Poha with Roasted Peanuts", "portion": 1.0, "unit": "plate",
            "carbs": 40.0, "fiber": 4.0, "protein": 7.0, "fat": 8.0, "digestion": 1.4,
            "pre": 117.0, "g15": 130.0, "g30": 146.0, "g45": 154.0, "g60": 150.0, "g120": 128.0,
            "peak": 154.0, "delta": 37.0, "verdict": "MODERATE", "intolerance": "TOLERATED",
            "notes": "Flattened by peanut fats, moderate excursion.", "pred": 150.0, "err": 4.0
        },
    ]

    for m in historical_meals:
        meal_time = now - timedelta(days=m["days_ago"], hours=(12 - m["hour"]))
        meal_record = {
            "timestamp": meal_time.isoformat(),
            "meal_type": m["meal_type"],
            "food_name": m["food_name"],
            "portion_amount": m["portion"],
            "portion_unit": m["unit"],
            "carbs_g": m["carbs"],
            "fiber_g": m["fiber"],
            "protein_g": m["protein"],
            "fat_g": m["fat"],
            "digestion_hours": m["digestion"],
            "pre_glucose": m["pre"],
            "glucose_15m": m["g15"],
            "glucose_30m": m["g30"],
            "glucose_45m": m["g45"],
            "glucose_60m": m["g60"],
            "glucose_120m": m["g120"],
            "peak_glucose": m["peak"],
            "delta_glucose": m["delta"],
            "verdict": m["verdict"],
            "intolerance_status": m["intolerance"],
            "intolerance_notes": m["notes"],
            "predicted_peak": m["pred"],
            "residual_error": m["err"],
        }
        meal_id = add_meal_log(patient_id, meal_record, db_path)

    # Save calibrated weights representing current learned state
    current_weights = {
        "carb_weight": 0.88,
        "fiber_weight": -0.48,
        "fat_weight": -0.16,
        "protein_weight": 0.08,
        "baseline_weight": 0.06,
        "hba1c_weight": 2.3,
        "bias": 4.8,
    }
    save_weights(
        patient_id=patient_id,
        weights=current_weights,
        step_info={
            "learning_step": len(historical_meals),
            "old_weights": DEFAULT_TWIN_WEIGHTS,
            "delta_weights": {k: round(current_weights[k] - DEFAULT_TWIN_WEIGHTS[k], 4) for k in current_weights},
            "loss": 4.2,
            "trigger_reason": "Seeded initial calibrated online learning state",
            "notes": "Trained across 22 historical patient meals with online residual learning.",
        },
        db_path=db_path
    )

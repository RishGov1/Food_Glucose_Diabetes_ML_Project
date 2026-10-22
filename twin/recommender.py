"""Food recommendation engine and glycemic tolerance categorization."""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from twin.config import FOOD_DATASET_PATH
from twin.database import get_food_history, get_latest_weights


def parse_digestion_hours(val: Any) -> float:
    """Parse numeric midpoint from text like '0.5-1.5 hrs' or '2-3 hrs'."""
    if pd.isna(val):
        return 1.5
    nums = re.findall(r"\d+(?:\.\d+)?", str(val))
    if not nums:
        return 1.5
    floats = [float(x) for x in nums]
    return sum(floats[:2]) / len(floats[:2])


def load_food_catalog(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and normalize the Indian Food nutrient catalog."""
    path = csv_path or FOOD_DATASET_PATH
    if not path.exists():
        # Fallback minimal dataset if file missing
        return pd.DataFrame([
            {"food_name": "Moong Dal Khichdi", "unit_serving_carb_g": 36.0, "unit_serving_fibre_g": 5.0, "unit_serving_protein_g": 12.0, "unit_serving_fat_g": 6.0, "servings_unit": "bowl", "estimated_digestion_time": "1.5-2 hrs", "digestion_hours": 1.8},
            {"food_name": "Palak Paneer", "unit_serving_carb_g": 8.0, "unit_serving_fibre_g": 4.5, "unit_serving_protein_g": 14.0, "unit_serving_fat_g": 18.0, "servings_unit": "cup", "estimated_digestion_time": "2-3 hrs", "digestion_hours": 2.5},
            {"food_name": "White Rice Biryani", "unit_serving_carb_g": 65.0, "unit_serving_fibre_g": 1.5, "unit_serving_protein_g": 9.0, "unit_serving_fat_g": 14.0, "servings_unit": "plate", "estimated_digestion_time": "2-3 hrs", "digestion_hours": 2.5},
            {"food_name": "Gulab Jamun", "unit_serving_carb_g": 52.0, "unit_serving_fibre_g": 0.5, "unit_serving_protein_g": 3.0, "unit_serving_fat_g": 12.0, "servings_unit": "piece", "estimated_digestion_time": "0.5-1 hrs", "digestion_hours": 0.8},
        ])

    df = pd.read_csv(path)
    # Ensure numeric columns
    for col in ["unit_serving_carb_g", "unit_serving_fibre_g", "unit_serving_protein_g", "unit_serving_fat_g", "unit_serving_freesugar_g", "unit_serving_energy_kcal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    df["food_name"] = df["food_name"].fillna("Unnamed Food").astype(str).str.strip()
    df["servings_unit"] = df["servings_unit"].fillna("serving").astype(str)
    df["digestion_hours"] = df["estimated_digestion_time"].apply(parse_digestion_hours)

    # Deduplicate
    df = df.drop_duplicates(subset=["food_name"]).sort_values("food_name").reset_index(drop=True)
    return df


def classify_food_item(
    food_row: Dict[str, Any],
    weights: Dict[str, float],
    portion: float = 1.0,
    historical_logs: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Classify food item into Allowed, Caution, or Not Recommended,
    factoring in the personal twin model weights and historical tolerance.
    """
    carbs = float(food_row.get("unit_serving_carb_g", 0.0)) * portion
    fiber = float(food_row.get("unit_serving_fibre_g", 0.0)) * portion
    protein = float(food_row.get("unit_serving_protein_g", 0.0)) * portion
    fat = float(food_row.get("unit_serving_fat_g", 0.0)) * portion
    sugar = float(food_row.get("unit_serving_freesugar_g", 0.0)) * portion
    digestion = float(food_row.get("digestion_hours", 1.5))
    calories = float(food_row.get("unit_serving_energy_kcal", 0.0)) * portion

    # Predicted postprandial rise according to patient's personal twin weights
    w_carb = float(weights.get("carb_weight", 0.85))
    w_fiber = float(weights.get("fiber_weight", -0.45))
    w_fat = float(weights.get("fat_weight", -0.15))
    w_protein = float(weights.get("protein_weight", 0.10))
    bias = float(weights.get("bias", 4.0))

    pred_rise = max(5.0, (w_carb * carbs) + (w_fiber * fiber) + (w_fat * fat) + (w_protein * protein) + bias)

    # Check historical intolerance records
    has_history = False
    past_avg_delta = None
    intolerance_flag = "UNKNOWN"
    intolerance_alert = False
    history_note = ""

    if historical_logs:
        has_history = True
        deltas = [r["delta_glucose"] for r in historical_logs]
        past_avg_delta = sum(deltas) / len(deltas)
        latest_status = historical_logs[-1].get("intolerance_status", "TOLERATED")
        intolerance_flag = latest_status

        if latest_status == "EMERGING_INTOLERANT":
            intolerance_alert = True
            history_note = f"⚠️ Emerging Intolerance: Recent meal triggered abnormal spike (+{deltas[-1]:.0f} mg/dL vs historical baseline {past_avg_delta:.0f} mg/dL)."
        elif latest_status == "KNOWN_INTOLERANT":
            history_note = f"🔴 Known Intolerance: Consistently induces high spikes (avg +{past_avg_delta:.0f} mg/dL)."
        elif past_avg_delta <= 32.0:
            history_note = f"🟢 High Historical Tolerance: Average rise only +{past_avg_delta:.0f} mg/dL across {len(deltas)} logged meals."

    # Verdict determination
    if intolerance_alert:
        tier = "NOT_RECOMMENDED"
        status_label = "⚠️ Emerging Intolerance"
        badge_color = "#E65100"  # Vibrant amber/orange warning
        reason = "Patient is becoming intolerant to this item; recent postprandial spikes show significant upward divergence from previous baseline."
    elif pred_rise > 52.0 or sugar > 22.0 or (past_avg_delta and past_avg_delta > 55.0):
        tier = "NOT_RECOMMENDED"
        status_label = "🔴 Not Recommended"
        badge_color = "#D32F2F"  # Red
        reason = f"High glycemic impact (predicted rise +{pred_rise:.0f} mg/dL). High rapid carb load."
    elif pred_rise > 35.0 or carbs > 42.0:
        tier = "CAUTION"
        status_label = "🟡 Caution / Portion Control"
        badge_color = "#F57F17"  # Yellow/Gold
        reason = f"Moderate glycemic impact (+{pred_rise:.0f} mg/dL rise). Best consumed in controlled portions or paired with green salad/protein."
    else:
        tier = "ALLOWED"
        status_label = "🟢 Allowed / Recommended"
        badge_color = "#2E7D32"  # Green
        reason = f"Optimal glycemic response (+{pred_rise:.0f} mg/dL rise). High fiber-to-carb ratio supports glucose stability."

    # Glycemic Tolerance Score (0 to 100, where 100 is best)
    tolerance_score = max(5.0, min(99.0, 100.0 - (pred_rise * 1.1) - (sugar * 0.8) + (fiber * 1.5)))
    if intolerance_alert:
        tolerance_score = min(tolerance_score, 35.0)

    return {
        "food_name": food_row.get("food_name", "Unknown"),
        "tier": tier,
        "status_label": status_label,
        "badge_color": badge_color,
        "reason": reason,
        "tolerance_score": round(tolerance_score, 1),
        "predicted_rise": round(pred_rise, 1),
        "carbs_g": round(carbs, 1),
        "fiber_g": round(fiber, 1),
        "protein_g": round(protein, 1),
        "fat_g": round(fat, 1),
        "sugar_g": round(sugar, 1),
        "calories_kcal": round(calories, 1),
        "digestion_hours": round(digestion, 1),
        "unit": food_row.get("servings_unit", "serving"),
        "portion": portion,
        "has_history": has_history,
        "history_note": history_note,
        "past_avg_delta": round(past_avg_delta, 1) if past_avg_delta else None,
        "intolerance_status": intolerance_flag,
    }


def get_recommendation_catalog(
    patient_id: str,
    tier_filter: str = "ALL",
    search_query: str = "",
    portion_multiplier: float = 1.0,
    db_path: Optional[Path] = None,
    limit: int = 150
) -> List[Dict[str, Any]]:
    """Generate categorized recommendations for the food catalog."""
    df_foods = load_food_catalog()
    weights = get_latest_weights(patient_id, db_path)

    results = []
    q = search_query.strip().lower()

    for _, row in df_foods.iterrows():
        food_name = row["food_name"]
        if q and q not in food_name.lower():
            continue

        # Look up history for this food
        history = get_food_history(patient_id, food_name, db_path)
        classification = classify_food_item(dict(row), weights, portion=portion_multiplier, historical_logs=history)

        if tier_filter == "ALL":
            results.append(classification)
        elif tier_filter == "ALLOWED" and classification["tier"] == "ALLOWED":
            results.append(classification)
        elif tier_filter == "CAUTION" and classification["tier"] == "CAUTION":
            results.append(classification)
        elif tier_filter == "NOT_RECOMMENDED" and classification["tier"] == "NOT_RECOMMENDED":
            results.append(classification)
        elif tier_filter == "INTOLERANCE_ALERTS" and classification["intolerance_status"] == "EMERGING_INTOLERANT":
            results.append(classification)

        if len(results) >= limit:
            break

    # Sort results: Allowed first, then Caution, then Not Recommended; sorted by tolerance score descending
    tier_order = {"ALLOWED": 0, "CAUTION": 1, "NOT_RECOMMENDED": 2}
    results.sort(key=lambda x: (tier_order.get(x["tier"], 3), -x["tolerance_score"]))
    return results

"""Digital Twin personalized predictive model and online adaptive learning engine."""
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from twin.config import WEIGHT_BOUNDS
from twin.database import get_latest_weights, save_weights


def predict_peak_rise(
    carbs: float,
    fiber: float,
    protein: float,
    fat: float,
    pre_glucose: float,
    hba1c: float,
    weights: Dict[str, float]
) -> float:
    """Predict peak glucose delta (rise over baseline) using current twin weights."""
    w_carb = float(weights.get("carb_weight", 0.85))
    w_fiber = float(weights.get("fiber_weight", -0.45))
    w_fat = float(weights.get("fat_weight", -0.15))
    w_protein = float(weights.get("protein_weight", 0.10))
    w_base = float(weights.get("baseline_weight", 0.05))
    w_hba1c = float(weights.get("hba1c_weight", 2.2))
    bias = float(weights.get("bias", 4.5))

    delta_hba1c = max(0.0, hba1c - 5.5)
    baseline_offset = max(0.0, pre_glucose - 100.0)

    rise = (
        (w_carb * carbs) +
        (w_fiber * fiber) +
        (w_fat * fat) +
        (w_protein * protein) +
        (w_base * baseline_offset) +
        (w_hba1c * delta_hba1c) +
        bias
    )
    return max(4.0, rise)


def generate_modeled_curve(
    pre_glucose: float,
    predicted_peak_delta: float,
    digestion_hours: float = 1.5,
    fat_g: float = 5.0,
    fiber_g: float = 3.0
) -> List[Tuple[int, float]]:
    """
    Generate a continuous 7-point postprandial curve (0, 15, 30, 45, 60, 90, 120 min)
    calibrated by gastrointestinal emptying kinetics.
    """
    # Fat and fiber delay gastric emptying time
    delay_factor = min(20.0, (fat_g * 0.4) + (fiber_g * 0.5))
    peak_minute = 40.0 + delay_factor + (digestion_hours * 5.0)
    peak_minute = max(30.0, min(75.0, peak_minute))

    curve = []
    for t in [0, 15, 30, 45, 60, 90, 120]:
        if t == 0:
            val = pre_glucose
        elif t <= peak_minute:
            # Rise phase (log-sigmoid rise)
            fraction = (t / peak_minute) ** 1.3
            val = pre_glucose + (predicted_peak_delta * fraction)
        else:
            # Clearance / recovery phase
            elapsed_after_peak = t - peak_minute
            clearance_half_life = 45.0 + (delay_factor * 0.5)
            decay = math.exp(-0.693 * (elapsed_after_peak / clearance_half_life))
            val = pre_glucose + (predicted_peak_delta * decay)
        curve.append((t, round(val, 1)))

    return curve


def adapt_twin_weights(
    patient_id: str,
    meal_record: Dict[str, Any],
    observed_peak: float,
    hba1c: float,
    current_weights: Optional[Dict[str, Any]] = None,
    learning_rate: float = 0.035,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Performs an online adaptive learning step using the post-meal observation.
    Updates twin weights, applies physiological constraints, logs telemetry, and persists to DB.
    """
    if current_weights is None:
        current_weights = get_latest_weights(patient_id, db_path)

    carbs = float(meal_record.get("carbs_g", 30.0))
    fiber = float(meal_record.get("fiber_g", 3.0))
    protein = float(meal_record.get("protein_g", 10.0))
    fat = float(meal_record.get("fat_g", 8.0))
    pre_glucose = float(meal_record.get("pre_glucose", 115.0))

    predicted_delta = predict_peak_rise(carbs, fiber, protein, fat, pre_glucose, hba1c, current_weights)
    predicted_peak = pre_glucose + predicted_delta
    residual_error = observed_peak - predicted_peak
    loss = round(residual_error ** 2, 2)

    # Normalized gradient step
    step_num = int(current_weights.get("learning_step", 0)) + 1
    # Modulate learning rate gently with experience (1 / sqrt(step))
    eff_lr = max(0.015, min(0.06, learning_rate * (12.0 / (10.0 + math.sqrt(step_num)))))

    # Gradients with feature scaling
    grad_carb = residual_error * (carbs / 50.0)
    grad_fiber = -residual_error * (fiber / 15.0)  # High error -> fiber was less protective than assumed
    grad_fat = -residual_error * (fat / 25.0)
    grad_protein = residual_error * (protein / 30.0)
    grad_bias = residual_error * 0.25

    # If this meal triggered an emerging intolerance alert, intensify the carb sensitivity adaptation
    is_intolerant = meal_record.get("intolerance_status") == "EMERGING_INTOLERANT"
    if is_intolerant:
        grad_carb *= 1.4
        grad_bias += 0.8

    new_weights = {}
    deltas = {}

    updates = {
        "carb_weight": grad_carb,
        "fiber_weight": grad_fiber,
        "fat_weight": grad_fat,
        "protein_weight": grad_protein,
        "baseline_weight": residual_error * 0.01,
        "hba1c_weight": residual_error * 0.05,
        "bias": grad_bias,
    }

    for k, grad in updates.items():
        curr_val = float(current_weights[k])
        # Regularized update: w_new = w_curr + lr * grad - L2_penalty
        delta = (eff_lr * grad)
        updated = curr_val + delta

        # Clamp within physiological bounds
        low_bound, high_bound = WEIGHT_BOUNDS[k]
        clamped = max(low_bound, min(high_bound, updated))
        new_weights[k] = round(clamped, 4)
        deltas[k] = round(clamped - curr_val, 4)

    # Telemetry notes
    notes = (
        f"Online step #{step_num}: residual error {residual_error:+.1f} mg/dL. "
        f"Carb weight {current_weights['carb_weight']:.2f} -> {new_weights['carb_weight']:.2f} ({deltas['carb_weight']:+.3f})."
    )
    if is_intolerant:
        notes += " [Emerging Intolerance boost applied]."

    step_info = {
        "learning_step": step_num,
        "meal_id": meal_record.get("id"),
        "old_weights": current_weights,
        "delta_weights": deltas,
        "loss": loss,
        "trigger_reason": f"Meal postprandial residual update: {meal_record.get('food_name', 'Meal')}",
        "notes": notes,
    }

    # Persist in DB
    weight_id = save_weights(patient_id, new_weights, step_info, db_path)

    return {
        "weight_id": weight_id,
        "learning_step": step_num,
        "old_weights": current_weights,
        "new_weights": new_weights,
        "deltas": deltas,
        "residual_error": round(residual_error, 2),
        "loss": loss,
        "notes": notes,
        "predicted_peak": round(predicted_peak, 1),
        "observed_peak": round(observed_peak, 1),
    }

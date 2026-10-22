"""Post-meal glycemic evaluation and intolerance detection engine."""
from pathlib import Path
from typing import Any, Dict, List, Optional

from twin.config import (
    INTOLERANCE_DELTA_JUMP,
    INTOLERANCE_HISTORICAL_BASELINE_MAX,
    INTOLERANCE_SPIKE_TRIGGER,
    SPIKE_GOOD_MAX,
    SPIKE_MODERATE_MAX,
    TARGET_MAX_GLUCOSE,
)
from twin.database import get_food_history


def evaluate_meal_response(
    pre_glucose: float,
    readings: Dict[str, Optional[float]],
    food_name: str,
    patient_id: str,
    predicted_peak: float,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluates postprandial glucose readings:
    1. Determines whether the response is GOOD, MODERATE, or BAD.
    2. Compares against the patient's personal history to detect emerging intolerance.
    """
    time_points = [
        (0, pre_glucose),
        (15, readings.get("glucose_15m")),
        (30, readings.get("glucose_30m")),
        (45, readings.get("glucose_45m")),
        (60, readings.get("glucose_60m")),
        (120, readings.get("glucose_120m")),
    ]

    valid_points = [(t, float(v)) for t, v in time_points if v is not None]
    if not valid_points:
        valid_points = [(0, pre_glucose)]

    # Peak glucose and timing
    peak_time, peak_glucose = max(valid_points, key=lambda x: x[1])
    delta_glucose = max(0.0, peak_glucose - pre_glucose)
    g120 = readings.get("glucose_120m")

    # 1. Glycemic Verdict (GOOD / MODERATE / BAD)
    if delta_glucose <= SPIKE_GOOD_MAX and peak_glucose <= TARGET_MAX_GLUCOSE and (g120 is None or g120 <= 140.0):
        verdict = "GOOD"
        verdict_badge = "🟢 GOOD — Optimal Glycemic Response"
        verdict_color = "#2E7D32"
        verdict_summary = (
            f"Your blood glucose rose by only +{delta_glucose:.0f} mg/dL (peak {peak_glucose:.0f} mg/dL at {peak_time}m). "
            f"Glucose remained comfortably within target boundaries without an adverse spike."
        )
    elif delta_glucose <= SPIKE_MODERATE_MAX and peak_glucose <= 180.0 and (g120 is None or g120 <= 170.0):
        verdict = "MODERATE"
        verdict_badge = "🟡 MODERATE — Borderline Glycemic Response"
        verdict_color = "#F57F17"
        verdict_summary = (
            f"Blood glucose experienced a moderate rise of +{delta_glucose:.0f} mg/dL (peak {peak_glucose:.0f} mg/dL at {peak_time}m). "
            f"Stayed below the acute threshold, but shows notable carbohydrate sensitivity."
        )
    else:
        verdict = "BAD"
        verdict_badge = "🔴 BAD — Glycemic Spike / Excursion"
        verdict_color = "#D32F2F"
        verdict_summary = (
            f"Significant glycemic spike detected (+{delta_glucose:.0f} mg/dL rise, peaking at {peak_glucose:.0f} mg/dL). "
            f"This exceeds recommended postprandial limits and causes high glycemic variability."
        )

    # 2. Historical Tolerance & Intolerance Analysis
    past_meals = get_food_history(patient_id, food_name, db_path)
    # Exclude entries that might share exact same current timestamp if already saved
    deltas = [m["delta_glucose"] for m in past_meals]

    intolerance_status = "TOLERATED"
    is_emerging_intolerant = False
    intolerance_title = "Normal Metabolic Tolerance"
    intolerance_message = ""

    if len(deltas) >= 2:
        past_avg_delta = sum(deltas) / len(deltas)
        past_max_delta = max(deltas)
        prior_verdicts = [m["verdict"] for m in past_meals]
        mostly_good_before = prior_verdicts.count("GOOD") >= len(prior_verdicts) / 2

        # Check for Emerging Intolerance:
        # Was previously well tolerated (past average delta <= INTOLERANCE_HISTORICAL_BASELINE_MAX or mostly good),
        # but current reading is high (>= INTOLERANCE_SPIKE_TRIGGER) and jumped by at least INTOLERANCE_DELTA_JUMP
        if (past_avg_delta <= INTOLERANCE_HISTORICAL_BASELINE_MAX or mostly_good_before) and (delta_glucose >= INTOLERANCE_SPIKE_TRIGGER) and (delta_glucose - past_avg_delta >= INTOLERANCE_DELTA_JUMP):
            intolerance_status = "EMERGING_INTOLERANT"
            is_emerging_intolerant = True
            intolerance_title = "⚠️ Emerging Intolerance Alert!"
            intolerance_color = "#E65100"
            jump = delta_glucose - past_avg_delta
            intolerance_message = (
                f"Developing Intolerance Detected! Historically, your Digital Twin recorded a safe average rise of "
                f"+{past_avg_delta:.1f} mg/dL across {len(deltas)} previous logs for '{food_name}'. "
                f"However, today's meal triggered a spike of +{delta_glucose:.1f} mg/dL (+{jump:.1f} mg/dL deviation). "
                f"Your physiological tolerance to this item is degrading. The Twin is updating its sensitivity weights and downgrading this food."
            )
        elif past_avg_delta > INTOLERANCE_HISTORICAL_BASELINE_MAX and delta_glucose > INTOLERANCE_HISTORICAL_BASELINE_MAX:
            intolerance_status = "KNOWN_INTOLERANT"
            intolerance_title = "🔴 Known Intolerance Confirmed"
            intolerance_color = "#D32F2F"
            intolerance_message = (
                f"Consistent glycemic sensitivity confirmed. '{food_name}' historically causes an average spike of "
                f"+{past_avg_delta:.1f} mg/dL. Today's rise (+{delta_glucose:.1f} mg/dL) remains in the intolerant zone."
            )
        else:
            intolerance_status = "TOLERATED"
            intolerance_title = "🟢 Stable Tolerance Confirmed"
            intolerance_color = "#2E7D32"
            intolerance_message = (
                f"Consistent positive tolerance maintained. Your postprandial rise (+{delta_glucose:.1f} mg/dL) aligns with "
                f"your historical baseline (+{past_avg_delta:.1f} mg/dL across {len(deltas)} meals)."
            )
    else:
        # First or second time logging this specific food
        if verdict == "BAD":
            intolerance_status = "MONITORING_SPIKE"
            intolerance_title = "⚠️ Elevated Initial Response"
            intolerance_color = "#F57F17"
            intolerance_message = f"First logs for '{food_name}' showed an elevated spike (+{delta_glucose:.1f} mg/dL). The Digital Twin will monitor future meals for intolerance patterns."
        else:
            intolerance_status = "TOLERATED"
            intolerance_title = "🟢 Favorable Baseline"
            intolerance_color = "#2E7D32"
            intolerance_message = f"Initial logs for '{food_name}' indicate good tolerance (+{delta_glucose:.1f} mg/dL rise)."

    # Residual error vs predicted peak
    residual_error = round(peak_glucose - predicted_peak, 2)

    return {
        "verdict": verdict,
        "verdict_badge": verdict_badge,
        "verdict_color": verdict_color,
        "verdict_summary": verdict_summary,
        "peak_glucose": round(peak_glucose, 1),
        "peak_time": peak_time,
        "delta_glucose": round(delta_glucose, 1),
        "recovery_120m": round(g120, 1) if g120 is not None else None,
        "intolerance_status": intolerance_status,
        "is_emerging_intolerant": is_emerging_intolerant,
        "intolerance_title": intolerance_title,
        "intolerance_color": intolerance_color,
        "intolerance_message": intolerance_message,
        "predicted_peak": round(predicted_peak, 1),
        "residual_error": residual_error,
        "valid_points": valid_points,
    }

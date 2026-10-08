import numpy as np
import pandas as pd

GLUCOSE_ONLY_FEATURES = ["g_120"]

COMMON_FEATURES = [
    "g_0",
    "g_120",
    "delta_120",
    "food_event_count",
]

OPTIONAL_MEAL_FEATURE_PREFIXES = [
    "meal_sugar",
    "meal_carb",
    "meal_protein",
    "meal_fat",
    "meal_fiber",
    "meal_calorie",
    "meal_kcal",
]

def choose_common_features(df):
    features = []
    for c in COMMON_FEATURES:
        if c in df.columns and df[c].notna().sum() > 0:
            features.append(c)
    for c in df.columns:
        if any(c.startswith(prefix) for prefix in OPTIONAL_MEAL_FEATURE_PREFIXES):
            if df[c].notna().sum() > 0:
                features.append(c)
    return features

def six_point_summary(values):
    offsets = np.array([0, 15, 40, 45, 60, 120], dtype=float)
    x = np.asarray(values, dtype=float)
    if x.shape[0] != 6:
        raise ValueError("Exactly six readings are required: 0, 15, 40, 45, 60, 120 minutes.")

    result = {}
    result["initial_glucose"] = float(x[0])
    result["two_hour_glucose"] = float(x[-1])
    result["peak_glucose"] = float(np.nanmax(x))
    result["peak_time_min"] = float(offsets[int(np.nanargmax(x))])
    result["peak_excursion"] = float(np.nanmax(x) - x[0])
    result["two_hour_excursion"] = float(x[-1] - x[0])
    result["auc_0_120"] = float(((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(x, offsets)))
    result["slope_0_15"] = float((x[1] - x[0]) / 15.0)
    result["slope_60_120"] = float((x[-1] - x[4]) / 60.0)
    result["recovery_60_120"] = float(x[4] - x[-1])
    result["pct_peak_rise"] = float(100 * (np.nanmax(x) - x[0]) / max(x[0], 1e-6))
    return result

import numpy as np
import pandas as pd

TIMEPOINTS = [0, 15, 40, 45, 60, 120]

# Features used by the D1NAMO post-meal response classifier.
# Food nutrients are deliberately not included unless they are observed and linked
# to each labelled D1NAMO meal; the separate Indian food table has no diagnosis labels.
MODEL_FEATURES = [
    "g_0", "g_15", "g_40", "g_45", "g_60", "g_120",
    "peak_glucose", "peak_time_min", "delta_120", "peak_excursion",
    "auc_0_120", "slope_0_15", "slope_60_120", "recovery_60_120",
    "pct_peak_rise",
]

GLUCOSE_ONLY_FEATURES = ["g_120"]


def six_point_summary(values):
    offsets = np.asarray(TIMEPOINTS, dtype=float)
    x = np.asarray(values, dtype=float)
    if x.shape[0] != 6:
        raise ValueError("Exactly six readings are required: 0, 15, 40, 45, 60, 120 minutes.")
    if not np.isfinite(x).all():
        raise ValueError("All six glucose readings must be provided as finite numbers.")

    peak_idx = int(np.argmax(x))
    auc_func = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    g0, g15, g60, g120 = x[0], x[1], x[4], x[5]
    return {
        "initial_glucose": float(g0),
        "two_hour_glucose": float(g120),
        "peak_glucose": float(x[peak_idx]),
        "peak_time_min": float(offsets[peak_idx]),
        "peak_excursion": float(x[peak_idx] - g0),
        "two_hour_excursion": float(g120 - g0),
        "delta_120": float(g120 - g0),
        "auc_0_120": float(auc_func(x, offsets)),
        "slope_0_15": float((g15 - g0) / 15.0),
        "slope_60_120": float((g120 - g60) / 60.0),
        "recovery_60_120": float(g60 - g120),
        "pct_peak_rise": float(100.0 * (x[peak_idx] - g0) / max(g0, 1e-6)),
    }


def prediction_frame(values, features=None):
    """Build the same named feature row used at training time."""
    x = np.asarray(values, dtype=float)
    if x.shape[0] != 6 or not np.isfinite(x).all():
        raise ValueError("Provide six finite glucose values in the order 0, 15, 40, 45, 60, 120 minutes.")
    row = {f"g_{minute}": float(value) for minute, value in zip(TIMEPOINTS, x)}
    row.update(six_point_summary(x))
    use = features or MODEL_FEATURES
    return pd.DataFrame([{name: row.get(name, np.nan) for name in use}], columns=use)


def choose_response_features(df):
    """Return supported glucose-response features in a stable, known order."""
    return [name for name in MODEL_FEATURES if name in df.columns]

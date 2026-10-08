from pathlib import Path
import json
import numpy as np
import pandas as pd

def load_config(path="config.yaml"):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def to_mg_dl(series):
    s = pd.to_numeric(series, errors="coerce")
    med = s.dropna().median()
    if pd.notna(med) and med < 30:
        return s * 18.0
    return s

def safe_auc(times, values):
    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    order = np.argsort(x[mask])
    return float(((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(y[mask][order], x[mask][order])))

def find_column(df, candidates):
    normalized = {
        str(c).strip().lower().replace(" ", "_").replace("-", "_"): c
        for c in df.columns
    }
    for cand in candidates:
        key = cand.lower().replace(" ", "_").replace("-", "_")
        if key in normalized:
            return normalized[key]
    for c in df.columns:
        key = str(c).lower().replace(" ", "_").replace("-", "_")
        if any(cand.lower().replace(" ", "_") in key for cand in candidates):
            return c
    return None

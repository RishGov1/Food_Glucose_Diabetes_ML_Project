from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

from .utils import find_column, to_mg_dl

TARGET_OFFSETS = [0, 15, 40, 45, 60, 120]

def infer_subject_id(path):
    parts = list(Path(path).parts)
    for p in reversed(parts):
        m = re.search(r"(?<!\d)(\d{3})(?!\d)", p)
        if m:
            return m.group(1)
    m = re.search(r"(?<!\d)(\d{3})(?!\d)", str(path))
    return m.group(1) if m else Path(path).parent.name

def infer_label(path):
    low = str(path).lower()
    if "healthy" in low:
        return 0
    if "diabetes" in low or "diabetic" in low:
        return 1
    return None

def parse_datetime(df):
    cols = {str(c).lower(): c for c in df.columns}
    datetime_col = None
    for c in ["datetime", "timestamp", "date_time", "time_stamp"]:
        if c in cols:
            datetime_col = cols[c]
            break
    if datetime_col is not None:
        return pd.to_datetime(df[datetime_col], errors="coerce", format="mixed")

    date_col = None
    time_col = None
    for c in ["date", "day"]:
        if c in cols:
            date_col = cols[c]
            break
    for c in ["time", "clock"]:
        if c in cols:
            time_col = cols[c]
            break
    if date_col is not None and time_col is not None:
        return pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str),
            errors="coerce",
        )
    if date_col is not None:
        return pd.to_datetime(df[date_col], errors="coerce")
    return pd.to_datetime(pd.Series([None] * len(df)), errors="coerce")

def parse_glucose_file(path):
    df = pd.read_csv(path)
    dt = parse_datetime(df)

    value_col = find_column(
        df,
        [
            "glucose", "glucose_value_mg_dl", "blood_glucose",
            "blood glucose", "bgl", "bg", "concentration", "value"
        ],
    )
    if value_col is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            # Prefer a column with a glucose-like name; otherwise first numeric column.
            value_col = numeric_cols[0]

    if value_col is None:
        raise ValueError(f"Could not identify a glucose column in {path}")

    out = pd.DataFrame({
        "timestamp": dt,
        "glucose": to_mg_dl(df[value_col]),
    }).dropna()

    out = out.sort_values("timestamp").drop_duplicates("timestamp")
    return out

def parse_food_file(path):
    df = pd.read_csv(path)
    dt = parse_datetime(df)
    out = pd.DataFrame({"timestamp": dt}).dropna()

    numeric = df.select_dtypes(include=np.number).copy()
    macro_map = {}
    for c in numeric.columns:
        key = str(c).lower().replace(" ", "_")
        if any(token in key for token in ["sugar", "carb", "protein", "fat", "fiber", "calorie", "kcal"]):
            macro_map[c] = numeric[c]
    if macro_map:
        for c, s in macro_map.items():
            out[str(c)] = pd.to_numeric(s.loc[out.index], errors="coerce").values

    return out.sort_values("timestamp").reset_index(drop=True)

def find_glucose_file(subject_dir):
    files = list(Path(subject_dir).rglob("*.csv"))
    for f in files:
        if f.name.lower() == "glucose.csv":
            return f
    for f in files:
        if "glucose" in f.name.lower():
            return f
    return None

def find_food_files(subject_dir):
    files = []
    for f in Path(subject_dir).rglob("*.csv"):
        low = f.name.lower()
        if "food" in low or "meal" in low:
            files.append(f)
    return files

def cluster_meals(food_df, gap_minutes=30):
    if food_df.empty:
        return pd.DataFrame(columns=["meal_time", "food_event_count"])

    times = food_df["timestamp"].sort_values().reset_index(drop=True)
    groups = (times.diff().dt.total_seconds().div(60).fillna(0) > gap_minutes).cumsum()

    rows = []
    for _, idx in times.groupby(groups).groups.items():
        t = times.loc[idx]
        rows.append({
            "meal_time": t.median(),
            "food_event_count": int(len(t)),
        })
    return pd.DataFrame(rows).sort_values("meal_time").reset_index(drop=True)

def nearest_value(glucose_df, target_time, tolerance_minutes):
    if glucose_df.empty or pd.isna(target_time):
        return np.nan

    idx = glucose_df["timestamp"].searchsorted(target_time)
    candidates = []
    if idx < len(glucose_df):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return np.nan

    best = min(
        candidates,
        key=lambda i: abs((glucose_df.iloc[i]["timestamp"] - target_time).total_seconds())
    )
    delta = abs((glucose_df.iloc[best]["timestamp"] - target_time).total_seconds()) / 60.0
    if delta <= tolerance_minutes:
        return float(glucose_df.iloc[best]["glucose"])
    return np.nan

def extract_response(glucose_df, meal_time, offsets, tolerance_minutes=8):
    vals = {}
    for m in offsets:
        vals[m] = nearest_value(
            glucose_df,
            meal_time + pd.Timedelta(minutes=int(m)),
            tolerance_minutes,
        )
    return vals

def add_response_features(row):
    g = [row.get(f"g_{m}") for m in TARGET_OFFSETS]
    finite = [(m, float(v)) for m, v in zip(TARGET_OFFSETS, g) if pd.notna(v)]

    if finite:
        peak_time, peak = max(finite, key=lambda x: x[1])
        row["peak_glucose"] = peak
        row["peak_time_min"] = peak_time
    else:
        row["peak_glucose"] = np.nan
        row["peak_time_min"] = np.nan

    g0 = row.get("g_0", np.nan)
    g120 = row.get("g_120", np.nan)

    row["delta_120"] = (
        g120 - g0 if pd.notna(g120) and pd.notna(g0) else np.nan
    )
    row["peak_excursion"] = (
        row["peak_glucose"] - g0
        if pd.notna(row["peak_glucose"]) and pd.notna(g0)
        else np.nan
    )

    # Trapezoidal AUC over the available six-point curve.
    if len(finite) >= 2:
        xs = np.array([x[0] for x in finite], dtype=float)
        ys = np.array([x[1] for x in finite], dtype=float)
        row["auc_0_120"] = float(((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(ys, xs)))
    else:
        row["auc_0_120"] = np.nan

    g15, g60 = row.get("g_15"), row.get("g_60")
    row["slope_0_15"] = (
        (g15 - g0) / 15.0 if pd.notna(g0) and pd.notna(g15) else np.nan
    )
    row["slope_60_120"] = (
        (g120 - g60) / 60.0 if pd.notna(g60) and pd.notna(g120) else np.nan
    )
    row["recovery_60_120"] = (
        g60 - g120 if pd.notna(g60) and pd.notna(g120) else np.nan
    )
    row["pct_peak_rise"] = (
        100.0 * (row["peak_glucose"] - g0) / g0
        if pd.notna(row["peak_glucose"]) and pd.notna(g0) and g0 > 0
        else np.nan
    )
    return row

def discover_subject_dirs(root):
    root = Path(root)
    candidates = set()
    for gf in root.rglob("*.csv"):
        if "glucose" in gf.name.lower():
            candidates.add(gf.parent)
    return sorted(candidates)

def load_d1namo(root, meal_gap_minutes=30, tolerance_minutes=8):
    root = Path(root)
    records = []
    for subject_dir in discover_subject_dirs(root):
        gf = find_glucose_file(subject_dir)
        if gf is None:
            continue

        label = infer_label(subject_dir)
        if label is None:
            warnings.warn(f"Could not infer label for {subject_dir}; skipping.")
            continue

        try:
            glucose = parse_glucose_file(gf)
        except Exception as exc:
            warnings.warn(f"Skipping {gf}: {exc}")
            continue

        food_files = find_food_files(subject_dir)
        food_frames = []
        for ff in food_files:
            try:
                fdf = parse_food_file(ff)
                if not fdf.empty:
                    food_frames.append(fdf)
            except Exception as exc:
                warnings.warn(f"Could not parse {ff}: {exc}")

        if not food_frames:
            continue

        food = pd.concat(food_frames, ignore_index=True).sort_values("timestamp")
        meals = cluster_meals(food, meal_gap_minutes)

        subject_id = infer_subject_id(subject_dir)

        for _, meal in meals.iterrows():
            row = {
                "subject_id": subject_id,
                "label": int(label),
                "meal_time": meal["meal_time"],
                "food_event_count": meal["food_event_count"],
                "source_dir": str(subject_dir),
            }

            response = extract_response(
                glucose,
                meal["meal_time"],
                TARGET_OFFSETS,
                tolerance_minutes,
            )
            for m, value in response.items():
                row[f"g_{m}"] = value

            # Add meal-level numeric metadata when available.
            window = food[
                (food["timestamp"] >= meal["meal_time"] - pd.Timedelta(minutes=15))
                & (food["timestamp"] <= meal["meal_time"] + pd.Timedelta(minutes=15))
            ]
            for col in food.columns:
                if col in {"timestamp"}:
                    continue
                if pd.api.types.is_numeric_dtype(food[col]):
                    value = pd.to_numeric(window[col], errors="coerce").sum(min_count=1)
                    if pd.notna(value):
                        safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(col).lower()).strip("_")
                        row[f"meal_{safe_name}"] = float(value)

            row = add_response_features(row)
            records.append(row)

    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError(
            "No D1NAMO meal samples were created. Check the D1NAMO extraction path and CSV layout."
        )
    return out

import numpy as np
import pandas as pd
from .forecast import ROOT

COLS = {"carb": "unit_serving_carb_g", "sugar": "unit_serving_freesugar_g",
        "fibre": "unit_serving_fibre_g", "protein": "unit_serving_protein_g", "fat": "unit_serving_fat_g"}


def load_food():
    df = pd.read_excel(ROOT / "data/food/indian_food.xlsx", sheet_name="Sheet1")
    df = df.dropna(subset=["food_name"]).drop_duplicates("food_name").copy()
    for col in COLS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def summarize(df, selected):
    totals = dict.fromkeys(COLS, 0.0)
    rows = []
    for name, quantity in selected:
        row = df.loc[df.food_name == name].iloc[0]
        item = {"Food": name, "Servings": quantity, "Serving unit": str(row.get("servings_unit", "Unspecified"))}
        missing = False
        for key, col in COLS.items():
            value = float(row[col]) * quantity
            item[key] = value
            if np.isfinite(value):
                totals[key] += value
            else:
                missing = True
        reasons = []
        if item["sugar"] >= 10:
            reasons.append("Free sugars ≥10 g")
        if item["carb"] >= 45:
            reasons.append("Carbohydrates ≥45 g")
        if item["carb"] >= 25 and item["fibre"] < 3:
            reasons.append("Carbohydrate portion with little fibre")
        item["Review"] = "; ".join(reasons) if reasons else "No configured nutrition flag"
        item["Incomplete nutrition"] = missing
        rows.append(item)
    return totals, pd.DataFrame(rows)

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib

sys.path.append(str(Path(__file__).resolve().parent))
from src.features import six_point_summary, prediction_frame
from src.utils import load_config

st.set_page_config(page_title="GlucoMeal AI", page_icon="🍽️", layout="wide")
st.title("🍽️ GlucoMeal AI")
st.caption("Food intake and post-meal glucose response analysis.")

model_path = Path("models/d1namo_response_model.joblib")
metadata_path = Path("outputs/response_model_metrics.json")
if not model_path.exists() or not metadata_path.exists():
    st.warning("Train the updated model first: python run_all.py")
    st.stop()
model = joblib.load(model_path)
metadata = pd.read_json(metadata_path, typ="series")
model_features = list(metadata["features"])

st.subheader("1. Enter glucose readings (mg/dL)")
st.caption("Enter readings at the stated time points using the same measurement protocol.")
cols = st.columns(6)
labels = [
    ("Immediately after", "g0"), ("15 minutes", "g15"), ("30 minutes", "g40"),
    ("45 minutes", "g45"), ("1 hour", "g60"), ("2 hours", "g120"),
]
values = []
for col, (label, key) in zip(cols, labels):
    with col:
        values.append(st.number_input(label, min_value=20.0, max_value=600.0,
                                      value=120.0, step=1.0, key=key))

st.subheader("2. Select the food consumed")
food_path = Path("data/food/indian_food.xlsx")
food_df = None
if food_path.exists():
    try:
        food_df = pd.read_excel(food_path, sheet_name="Sheet1")
        food_df["food_name"] = food_df["food_name"].astype(str)
        food_df = food_df.dropna(subset=["food_name"]).drop_duplicates("food_name")
    except Exception as exc:
        st.error(f"Could not load food database: {exc}")
else:
    st.warning("Food database not found at data/food/indian_food.xlsx.")

chosen_foods = []
if food_df is not None and not food_df.empty:
    food_names = sorted(food_df["food_name"].unique().tolist())
    chosen = st.multiselect("Food items (select all items in the meal)", options=food_names,
                            help="Nutrition values are estimates from the supplied spreadsheet.")
    for food_name in chosen:
        row = food_df.loc[food_df["food_name"] == food_name].iloc[0]
        unit = str(row.get("servings_unit", "serving"))
        qty = st.number_input(f"{food_name} — quantity ({unit})", min_value=0.25,
                              max_value=20.0, value=1.0, step=0.25,
                              key=f"qty_{row.get('food_code', food_name)}")
        chosen_foods.append((row, float(qty)))

if st.button("Analyze glucose and meal", type="primary"):
    summary = six_point_summary(values)
    x = prediction_frame(values, model_features)
    probability = float(model.predict_proba(x)[0, 1])
    prediction = int(probability >= 0.5)

    st.subheader("3. ML glucose-response estimate")
    a, b, c = st.columns(3)
    a.metric("Model-estimated class probability", f"{probability * 100:.1f}%")
    b.metric("2-hour glucose", f"{values[-1]:.1f} mg/dL")
    c.metric("Peak glucose", f"{summary['peak_glucose']:.1f} mg/dL")
    if prediction:
        st.warning("Model classification: higher likelihood of belonging to the Type 1 diabetes cohort.")
    else:
        st.success("Model classification: higher likelihood of belonging to the healthy cohort.")
    st.caption("This is a small-cohort research classification, not an individual clinical diagnosis or calibrated clinical probability.")

    st.subheader("4. Post-meal glucose response")
    metrics = [
        ("Peak time", f"{summary['peak_time_min']:.0f} min"),
        ("Peak excursion", f"{summary['peak_excursion']:.1f} mg/dL"),
        ("2-hour change from initial", f"{summary['two_hour_excursion']:.1f} mg/dL"),
        ("AUC (0–120 min)", f"{summary['auc_0_120']:.0f} mg·min/dL"),
        ("0–15 min slope", f"{summary['slope_0_15']:.2f} mg/dL/min"),
        ("60–120 min slope", f"{summary['slope_60_120']:.2f} mg/dL/min"),
        ("60–120 min recovery", f"{summary['recovery_60_120']:.1f} mg/dL"),
        ("Peak rise", f"{summary['pct_peak_rise']:.1f}%"),
    ]
    grid = st.columns(4)
    for i, (name, value) in enumerate(metrics):
        grid[i % 4].metric(name, value)
    chart = pd.DataFrame({"Minutes after meal": [0, 15, 40, 45, 60, 120],
                          "Measured glucose": values}).set_index("Minutes after meal")
    st.line_chart(chart)

    st.subheader("5. Food nutrition and meal context")
    if chosen_foods:
        totals = {"Energy (kcal)": 0.0, "Carbohydrates (g)": 0.0,
                  "Free sugars (g)": 0.0, "Protein (g)": 0.0,
                  "Fat (g)": 0.0, "Fibre (g)": 0.0}
        details = []
        mapping = [
            ("unit_serving_energy_kcal", "Energy (kcal)"),
            ("unit_serving_carb_g", "Carbohydrates (g)"),
            ("unit_serving_freesugar_g", "Free sugars (g)"),
            ("unit_serving_protein_g", "Protein (g)"),
            ("unit_serving_fat_g", "Fat (g)"),
            ("unit_serving_fibre_g", "Fibre (g)"),
        ]
        for row, qty in chosen_foods:
            item = {"Food": row["food_name"], "Servings": qty,
                    "Serving unit": str(row.get("servings_unit", "serving"))}
            for source, label in mapping:
                val = pd.to_numeric(row.get(source, 0), errors="coerce")
                val = 0.0 if pd.isna(val) else float(val)
                item[label] = val * qty
                totals[label] += val * qty
            item["Estimated digestion time"] = str(row.get("estimated_digestion_time", "Not available"))
            details.append(item)
        st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)
        st.write("**Estimated meal totals**")
        tcols = st.columns(3)
        for i, (label, val) in enumerate(totals.items()):
            tcols[i % 3].metric(label, f"{val:.1f}")

        st.write("**Food + measured-response interpretation**")
        carb = totals["Carbohydrates (g)"]
        sugar = totals["Free sugars (g)"]
        fibre = totals["Fibre (g)"]
        if carb > 0:
            st.write(f"This selected meal is estimated to contain {carb:.1f} g carbohydrates, {sugar:.1f} g free sugars, and {fibre:.1f} g fibre.")
            st.write(f"The observed peak excursion was {summary['peak_excursion']:.1f} mg/dL and the 2-hour change was {summary['two_hour_excursion']:.1f} mg/dL.")
            st.write("Use these together as meal-response context: the glucose curve is the measured response, while the food database is an estimated composition. The spreadsheet does not contain paired glucose outcomes for these exact foods, so it cannot establish a food-specific optimal curve or a learned food effect on diabetes probability.")
    else:
        st.write("No food items selected. Select the foods and servings consumed for nutrition context.")

    st.subheader("6. General reference comparison")
    st.write("For many non-pregnant adults who already have diabetes, a commonly used management target is peak post-meal glucose below 180 mg/dL, measured 1–2 hours after beginning a meal. This is a management target, not a diagnostic threshold or a personalized optimal value.")
    for label, value in [("1-hour reading", values[4]), ("2-hour reading", values[5])]:
        if value < 180:
            st.success(f"{label}: {value:.1f} mg/dL — below the general 180 mg/dL management reference.")
        else:
            st.warning(f"{label}: {value:.1f} mg/dL — at or above the general 180 mg/dL management reference.")
    st.warning("D1NAMO labels compare healthy participants with people who have Type 1 diabetes. PIMA's glucose value is from a 2-hour OGTT, not a six-point meal curve. Neither dataset validates this as a clinical diagnostic tool. Seek clinical testing for diagnosis.")

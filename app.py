import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "demo_model.joblib"
FOOD_PATH = ROOT / "data" / "indian_food.csv"

st.set_page_config(page_title="Food + Glucose ML Demo", page_icon="🧪", layout="centered")
st.title("Food-Aware Post-Meal Glucose ML Demo")
st.warning(
    "EDUCATIONAL PROTOTYPE ONLY — this is not a Type 1 diabetes diagnostic tool. "
    "The model was trained on synthetic examples and invented labels, not clinical patient data. "
    "Do not use the output for medical decisions."
)
st.write(
    "This updated version uses both the six glucose readings and nutritional features of the selected food "
    "(estimated carbohydrates, fibre, fat, protein, and digestion time)."
)

def digestion_midpoint(value):
    if pd.isna(value):
        return 1.5
    nums = re.findall(r"\d+(?:\.\d+)?", str(value))
    if not nums:
        return 1.5
    vals = [float(x) for x in nums]
    return sum(vals[:2]) / len(vals[:2]) if len(vals) >= 2 else vals[0]

@st.cache_data
def load_foods():
    foods = pd.read_csv(FOOD_PATH)
    for c in ["unit_serving_carb_g", "unit_serving_fibre_g", "unit_serving_fat_g", "unit_serving_protein_g"]:
        foods[c] = pd.to_numeric(foods[c], errors="coerce").fillna(0)
    foods["digestion_hours_numeric"] = foods["estimated_digestion_time"].apply(digestion_midpoint)
    foods["food_name"] = foods["food_name"].fillna("Unnamed food").astype(str)
    foods = foods.drop_duplicates(subset=["food_name"]).sort_values("food_name")
    return foods.reset_index(drop=True)

foods = load_foods()

with st.expander("Important limitation"):
    st.markdown(
        "- The model output is a synthetic-pattern class, not the probability of Type 1 diabetes.\n"
        "- Type 1 diabetes cannot be diagnosed or ruled out from these six readings and a food selection.\n"
        "- The food dataset's nutrition and digestion values are estimates; they do not fully represent a mixed meal or an individual's digestion.\n"
        "- Real clinical use would require consented, representative data with verified diagnoses, patient-level validation, external testing, and clinician oversight."
    )

st.subheader("1. Select the food consumed")
food_name = st.selectbox("Food item", foods["food_name"].tolist(), index=0)
food_row = foods.loc[foods["food_name"] == food_name].iloc[0]
unit = str(food_row.get("servings_unit", "serving"))
servings = st.number_input(f"Number of servings ({unit})", min_value=0.25, max_value=10.0, value=1.0, step=0.25)

carbs = float(food_row["unit_serving_carb_g"]) * servings
fibre = float(food_row["unit_serving_fibre_g"]) * servings
fat = float(food_row["unit_serving_fat_g"]) * servings
protein = float(food_row["unit_serving_protein_g"]) * servings
digestion = float(food_row["digestion_hours_numeric"])

with st.container(border=True):
    st.caption("Estimated nutrition for the selected portion")
    a, b = st.columns(2)
    a.metric("Carbohydrates", f"{carbs:.1f} g")
    b.metric("Fibre", f"{fibre:.1f} g")
    c, d = st.columns(2)
    c.metric("Fat", f"{fat:.1f} g")
    d.metric("Protein", f"{protein:.1f} g")
    st.write(f"Dataset digestion-time estimate: **{food_row.get('estimated_digestion_time', 'Not available')}**")
    st.caption("These are dataset estimates scaled by serving count, not personalized physiological measurements.")

st.subheader("2. Enter glucose readings")
fields = [
    ("glucose_0min", "Immediately after food (mg/dL)", 100.0),
    ("glucose_15min", "15 minutes (mg/dL)", 115.0),
    ("glucose_30min", "30 minutes (mg/dL)", 130.0),
    ("glucose_45min", "45 minutes (mg/dL)", 125.0),
    ("glucose_60min", "60 minutes (mg/dL)", 110.0),
    ("glucose_120min", "120 minutes (mg/dL)", 100.0),
]
with st.form("readings"):
    cols = st.columns(2)
    readings = {}
    for i, (key, label, default) in enumerate(fields):
        with cols[i % 2]:
            readings[key] = st.number_input(label, min_value=20.0, max_value=600.0, value=default, step=1.0)
    submitted = st.form_submit_button("Run food-aware educational demo")

if submitted:
    feature_row = {
        **readings,
        "meal_carbs_g": carbs,
        "meal_fibre_g": fibre,
        "meal_fat_g": fat,
        "meal_protein_g": protein,
        "digestion_hours": digestion,
    }
    feature_order = [
        "glucose_0min", "glucose_15min", "glucose_30min", "glucose_45min", "glucose_60min", "glucose_120min",
        "meal_carbs_g", "meal_fibre_g", "meal_fat_g", "meal_protein_g", "digestion_hours"
    ]
    X = pd.DataFrame([feature_row], columns=feature_order)
    model = joblib.load(MODEL_PATH)
    pred = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    st.subheader("Synthetic-pattern output")
    st.write(f"Demo class: **{pred}**")
    st.write(f"Model score for synthetic class 1: **{probs[1] * 100:.1f}%**")
    st.caption("This score is not the probability that you have Type 1 diabetes. Class 0 does not confirm health, and class 1 does not diagnose diabetes.")
    st.info(
        "If you are concerned about diabetes or have symptoms, contact a healthcare professional. "
        "Seek urgent care for severe symptoms such as vomiting, abdominal pain, deep/rapid breathing, confusion, or marked drowsiness."
    )

with st.expander("Food dataset browser"):
    search = st.text_input("Search foods", placeholder="e.g., dosa, rice, tea")
    view = foods
    if search:
        view = view[view["food_name"].str.contains(search, case=False, na=False)]
    display_cols = [c for c in [
        "food_name", "servings_unit", "unit_serving_carb_g", "unit_serving_fibre_g",
        "unit_serving_fat_g", "unit_serving_protein_g", "estimated_digestion_time"
    ] if c in view.columns]
    st.dataframe(view[display_cols].head(100), use_container_width=True)

import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "demo_model.joblib"
IMPACT_MODEL_PATH = ROOT / "models" / "food_impact_model.joblib"
FOOD_PATH = ROOT / "data" / "indian_food.csv"

st.set_page_config(page_title="Glucose Digital Twin — Educational Demo", page_icon="🧬", layout="wide")
st.title("🧬 Food-Aware Glucose Digital Twin")
st.caption("Detailed combined output from the food-impact and glucose-pattern models")
st.warning(
    "EDUCATIONAL PROTOTYPE ONLY: both models were trained on synthetic data with invented labels. "
    "Their scores are not clinical probabilities, a diagnosis, or a validated forecast of an individual's glucose response. "
    "Do not use this app to change medication, insulin, food intake, or treatment."
)
st.write(
    "Select a food and portion, enter current glucose and HbA1c, and enter post-meal readings if available. "
    "The app combines both model outputs with a plain-language explanation of the recorded glucose pattern and meal nutrients."
)


def digestion_midpoint(value):
    nums = re.findall(r"\d+(?:\.\d+)?", str(value)) if pd.notna(value) else []
    vals = [float(x) for x in nums]
    return sum(vals[:2]) / len(vals[:2]) if len(vals) >= 2 else (vals[0] if vals else 1.5)


@st.cache_data
def load_foods():
    foods = pd.read_csv(FOOD_PATH)
    for c in ["unit_serving_carb_g", "unit_serving_fibre_g", "unit_serving_fat_g", "unit_serving_protein_g"]:
        foods[c] = pd.to_numeric(foods[c], errors="coerce").fillna(0)
    foods["digestion_hours_numeric"] = foods["estimated_digestion_time"].apply(digestion_midpoint)
    foods["food_name"] = foods["food_name"].fillna("Unnamed food").astype(str)
    return foods.drop_duplicates(subset=["food_name"]).sort_values("food_name").reset_index(drop=True)


@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH), joblib.load(IMPACT_MODEL_PATH)


foods = load_foods()
with st.expander("What this prototype can and cannot tell you"):
    st.markdown(
        "- **Food-impact model:** classifies an invented synthetic label using current glucose, HbA1c, meal nutrients and estimated digestion time. Its score is not a real-world probability.\n"
        "- **Glucose-pattern model:** classifies a synthetic pattern from the six entered readings and related features. It is not a diabetes classifier.\n"
        "- **Combined explanation:** summarizes whether the two model outputs agree, describes the readings you entered, and gives general nutrition context. It does not create a clinically validated glucose forecast.\n"
        "- **HbA1c:** reflects longer-term average glucose (roughly the previous 2–3 months); one meal does not meaningfully change HbA1c immediately.\n"
        "- Actual meal responses vary with portion size, food preparation, mixed meals, medication/insulin, activity, stress, illness and individual physiology."
    )

st.header("1. Food and portion")
food_name = st.selectbox("Food item", foods["food_name"].tolist())
food_row = foods.loc[foods["food_name"] == food_name].iloc[0]
unit = str(food_row.get("servings_unit", "serving"))
servings = st.number_input(f"Number of servings ({unit})", min_value=0.25, max_value=10.0, value=1.0, step=0.25)
carbs = float(food_row["unit_serving_carb_g"]) * servings
fibre = float(food_row["unit_serving_fibre_g"]) * servings
fat = float(food_row["unit_serving_fat_g"]) * servings
protein = float(food_row["unit_serving_protein_g"]) * servings
digestion = float(food_row["digestion_hours_numeric"])
with st.container(border=True):
    st.caption("Estimated nutrients for your selected portion, scaled from the supplied food dataset")
    a, b, c, d = st.columns(4)
    a.metric("Carbohydrates", f"{carbs:.1f} g")
    b.metric("Fibre", f"{fibre:.1f} g")
    c.metric("Fat", f"{fat:.1f} g")
    d.metric("Protein", f"{protein:.1f} g")
    st.write(f"Dataset digestion-time estimate: **{food_row.get('estimated_digestion_time', 'Not available')}**")

st.header("2. Current measurements")
x, y = st.columns(2)
with x:
    current_glucose = st.number_input("Current blood glucose (mg/dL)", min_value=20.0, max_value=600.0, value=120.0, step=1.0)
with y:
    hba1c = st.number_input("Most recent HbA1c (%)", min_value=3.0, max_value=20.0, value=6.5, step=0.1)

st.header("3. Glucose readings around the meal")
st.caption("Enter measured values if you have them. Defaults are example values only; they are not predictions.")
fields = [
    ("glucose_0min", "Immediately / baseline (mg/dL)", current_glucose),
    ("glucose_15min", "15 minutes (mg/dL)", current_glucose + 15),
    ("glucose_30min", "30 minutes (mg/dL)", current_glucose + 25),
    ("glucose_45min", "45 minutes (mg/dL)", current_glucose + 20),
    ("glucose_60min", "60 minutes (mg/dL)", current_glucose + 10),
    ("glucose_120min", "120 minutes (mg/dL)", current_glucose),
]
with st.form("readings"):
    cols = st.columns(3)
    readings = {}
    for i, (key, label, default) in enumerate(fields):
        with cols[i % 3]:
            readings[key] = st.number_input(label, min_value=20.0, max_value=600.0, value=float(default), step=1.0, key=key)
    submitted = st.form_submit_button("Generate detailed combined report", type="primary", use_container_width=True)

if submitted:
    impact_features = ["current_glucose_mgdl", "hba1c_percent", "meal_carbs_g", "meal_fibre_g", "meal_fat_g", "meal_protein_g", "digestion_hours"]
    impact_row = pd.DataFrame([{
        "current_glucose_mgdl": current_glucose,
        "hba1c_percent": hba1c,
        "meal_carbs_g": carbs,
        "meal_fibre_g": fibre,
        "meal_fat_g": fat,
        "meal_protein_g": protein,
        "digestion_hours": digestion,
    }], columns=impact_features)

    main_features = ["glucose_0min", "glucose_15min", "glucose_30min", "glucose_45min", "glucose_60min", "glucose_120min",
                     "current_glucose_mgdl", "hba1c_percent", "meal_carbs_g", "meal_fibre_g", "meal_fat_g", "meal_protein_g", "digestion_hours"]
    pattern_row = {**readings, "current_glucose_mgdl": current_glucose, "hba1c_percent": hba1c,
                   "meal_carbs_g": carbs, "meal_fibre_g": fibre, "meal_fat_g": fat,
                   "meal_protein_g": protein, "digestion_hours": digestion}
    pattern_X = pd.DataFrame([pattern_row], columns=main_features)
    pattern_model, impact_model = load_models()
    impact_prob = float(impact_model.predict_proba(impact_row)[0][1])
    pattern_prob = float(pattern_model.predict_proba(pattern_X)[0][1])
    pattern_pred = int(pattern_prob >= 0.5)

    if impact_prob >= 0.60:
        impact_label = "Higher synthetic modeled food-impact category"
        impact_explanation = "The model's score is at or above its 60% threshold, so it assigns the higher synthetic category. This is a learned pattern in invented data, not a measured or clinically validated effect."
    elif impact_prob <= 0.40:
        impact_label = "Lower synthetic modeled food-impact category"
        impact_explanation = "The model's score is at or below its 40% threshold, so it assigns the lower synthetic category. This does not guarantee a small glucose rise for a real person."
    else:
        impact_label = "Uncertain / middle synthetic category"
        impact_explanation = "The model's score falls between 40% and 60%, so the app reports a middle category rather than a clear lower or higher result."

    if pattern_pred == 1:
        pattern_label = "Class 1 — higher/later-rise-like synthetic pattern"
        pattern_explanation = "The model assigns more than or equal to 50% to synthetic class 1. In this generated dataset, class 1 is associated with a more elevated/later glucose pattern; it does not mean diabetes."
    else:
        pattern_label = "Class 0 — lower/recovery-like synthetic pattern"
        pattern_explanation = "The model assigns less than 50% to synthetic class 1 and therefore selects class 0. In this generated dataset, class 0 is associated with a comparatively lower/recovery-like pattern; it does not rule out diabetes."

    values = [readings[k] for k, _, _ in fields]
    times = [0, 15, 30, 45, 60, 120]
    peak_idx = max(range(len(values)), key=lambda i: values[i])
    peak = values[peak_idx]
    peak_time = times[peak_idx]
    baseline = values[0]
    two_hour = values[-1]
    peak_change = peak - baseline
    two_hour_change = two_hour - baseline
    change_word = "increased" if peak_change > 0 else ("decreased" if peak_change < 0 else "did not change")
    recovery_word = "below" if two_hour < peak else ("above" if two_hour > peak else "equal to")

    st.header("4. Combined interpretation")
    left, right = st.columns(2)
    with left:
        st.subheader("Food-impact model")
        st.metric("Higher-category score", f"{impact_prob * 100:.1f}%")
        st.write(f"**Result:** {impact_label}")
        st.write(impact_explanation)
    with right:
        st.subheader("Glucose-pattern model")
        st.metric("Class 1 score", f"{pattern_prob * 100:.1f}%", help="Score for the synthetic class 1 label, not a diabetes probability")
        st.write(f"**Result:** {pattern_label}")
        st.write(pattern_explanation)

    st.subheader("What the two outputs mean together")
    if impact_prob >= 0.60 and pattern_pred == 1:
        combined = "Both models lean toward their higher/elevated synthetic categories."
        combined_detail = "The food-impact model selected its higher category, and the glucose-pattern model selected its higher/later-rise-like class. This is agreement between two synthetic models, not proof that this meal will cause a specific rise."
    elif impact_prob <= 0.40 and pattern_pred == 0:
        combined = "Both models lean toward their lower synthetic categories."
        combined_detail = "The food-impact model selected its lower category, and the glucose-pattern model selected its lower/recovery-like class. This is not a guarantee of a small real-world response or absence of diabetes."
    else:
        combined = "The two models give a mixed or less decisive picture."
        combined_detail = "The models do not both select matching lower or higher categories. Consider their different inputs and synthetic labels; do not treat either score as a medical conclusion."
    st.info(f"**Combined result: {combined}**\n\n{combined_detail}")

    st.subheader("What your entered glucose readings show")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Entered baseline", f"{baseline:.0f} mg/dL")
    m2.metric("Highest entered value", f"{peak:.0f} mg/dL", f"{peak_change:+.0f} mg/dL vs baseline")
    m3.metric("Time of entered peak", f"{peak_time} min")
    m4.metric("Entered 120-min value", f"{two_hour:.0f} mg/dL", f"{two_hour_change:+.0f} mg/dL vs baseline")
    chart_df = pd.DataFrame({"Minutes after meal": times, "Entered glucose (mg/dL)": values}).set_index("Minutes after meal")
    st.line_chart(chart_df, use_container_width=True)
    st.caption("This chart plots the readings you entered; it is not a model-generated forecast.")
    st.write(f"Your entered values {change_word} from baseline to a peak of **{peak:.0f} mg/dL at {peak_time} minutes**. At 120 minutes, the value is {recovery_word} the recorded peak. These are descriptions of the entered numbers, not a clinical interpretation by themselves.")

    st.subheader("How this selected food may influence glucose — general nutrition context")
    nutrition_notes = []
    if carbs >= 60:
        nutrition_notes.append(f"**Carbohydrates ({carbs:.1f} g):** this is a relatively carbohydrate-heavy portion. Digestible carbohydrates are generally the meal component most directly associated with a rise in blood glucose, though the size and timing vary by person and meal.")
    elif carbs >= 25:
        nutrition_notes.append(f"**Carbohydrates ({carbs:.1f} g):** this portion contains a moderate amount of carbohydrate, which can raise glucose after eating. The actual response depends on the person and the rest of the meal.")
    else:
        nutrition_notes.append(f"**Carbohydrates ({carbs:.1f} g):** the dataset estimates a lower carbohydrate amount for this portion than a higher-carb meal. It can still affect glucose; this estimate alone cannot predict the size of the response.")
    if fibre >= 5:
        nutrition_notes.append(f"**Fibre ({fibre:.1f} g):** fibre can slow digestion and carbohydrate absorption for some meals, potentially moderating or delaying the rise.")
    else:
        nutrition_notes.append(f"**Fibre ({fibre:.1f} g):** the dataset estimates {fibre:.1f} g for this portion. Fibre amount and food structure can affect the speed of absorption; the value here is an estimate from the supplied dataset.")
    if fat >= 15 or protein >= 20:
        nutrition_notes.append(f"**Fat and protein ({fat:.1f} g fat, {protein:.1f} g protein):** larger amounts can slow stomach emptying or delay the timing of glucose absorption in mixed meals, so the response may be later or more prolonged for some people. This does not necessarily mean the total rise will be lower.")
    else:
        nutrition_notes.append(f"**Fat and protein ({fat:.1f} g fat, {protein:.1f} g protein):** these nutrients also contribute to the mixed-meal response, but their effects depend on the full meal and individual physiology.")
    nutrition_notes.append(f"**Dataset digestion-time estimate ({food_row.get('estimated_digestion_time', 'not available')}):** this is a general food-dataset estimate, not a measured time for your body and not a glucose prediction.")
    for note in nutrition_notes:
        st.markdown(f"- {note}")

    st.subheader("What about HbA1c?")
    st.write(
        f"The entered HbA1c is **{hba1c:.1f}%**. HbA1c reflects average glucose over roughly the previous 2–3 months; a single meal does not immediately change it in a meaningful way. "
        "This app does not predict a future HbA1c value. The food-impact model uses the current HbA1c as one input, but its score is based on synthetic labels and must not be interpreted as a medical risk estimate."
    )
    if hba1c < 5.7:
        st.write("For general reference, an HbA1c below 5.7% is commonly considered below the prediabetes range in standard adult screening guidance. A single result is not a complete assessment and interpretation can vary with medical circumstances.")
    elif hba1c < 6.5:
        st.write("For general reference, an HbA1c from 5.7% to 6.4% is commonly considered the prediabetes range in standard adult screening guidance. Discuss results with a healthcare professional; this app cannot diagnose a condition.")
    else:
        st.write("For general reference, an HbA1c of 6.5% or higher is in the diabetes range used in standard adult diagnostic criteria, but diagnosis generally requires appropriate clinical assessment and confirmation. Please discuss this result with a healthcare professional.")

    st.subheader("Plain-language summary")
    st.markdown(
        f"- **Meal:** {food_name}, {servings:g} {unit}(s), estimated {carbs:.1f} g carbohydrate and {fibre:.1f} g fibre.\n"
        f"- **Model combination:** {combined}\n"
        f"- **Recorded readings:** entered peak {peak:.0f} mg/dL at {peak_time} minutes; entered 120-minute value {two_hour:.0f} mg/dL.\n"
        f"- **HbA1c:** {hba1c:.1f}%; this is a longer-term marker, not a value that this app predicts will change after the meal.\n"
        "- **Limit:** no real individual glucose forecast or Type 1 diabetes diagnosis is produced by this synthetic prototype."
    )
    st.caption("If these are actual measurements and you are concerned about them, contact a qualified healthcare professional. Seek urgent care for high glucose accompanied by vomiting, abdominal pain, rapid/deep breathing, or confusion.")

with st.expander("Browse the food dataset"):
    search = st.text_input("Search food items", placeholder="e.g., dosa, rice, tea")
    view = foods
    if search:
        view = view[view["food_name"].str.contains(search, case=False, na=False)]
    cols_show = [c for c in ["food_name", "servings_unit", "unit_serving_carb_g", "unit_serving_fibre_g", "unit_serving_fat_g", "unit_serving_protein_g", "estimated_digestion_time"] if c in view.columns]
    st.dataframe(view[cols_show].head(100), use_container_width=True)

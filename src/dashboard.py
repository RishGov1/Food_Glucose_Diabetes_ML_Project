from datetime import datetime, timedelta, timezone
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.forecast import ROOT
from src.features import prediction_frame, TIMEPOINTS
from src.food import load_food, summarize
from src.personal import profile_key, history, correction, save_observation

st.set_page_config(page_title="GlucoMeal · Personal glucose twin", page_icon="◒", layout="wide")
st.title("GlucoMeal")
st.caption("Personal glucose twin · meal tracking · local learning")
st.info("Research prototype. Forecasts are experimental, especially outside the small Type 1 diabetes training cohort. Do not use estimates to diagnose diabetes or change insulin or medication.")
if not (ROOT / "models/twin_forecast.joblib").exists():
    st.warning("Training is not finished. Run .venv\\Scripts\\python.exe run_all.py and refresh.")
    st.stop()


@st.cache_data
def foods():
    return load_food()


@st.cache_resource
def neural_device(device):
    import openvino as ov
    return ov.Core().compile_model(str(ROOT / "models/meal_neural.xml"), device)


bundle = joblib.load(ROOT / "models/twin_forecast.joblib")
report = json.loads((ROOT / "outputs/twin_metrics.json").read_text())
if report["holdout"]["mae"] > report["holdout_persistence"]["mae"]:
    st.warning("Validation limitation: the selected ML model did not beat the last-reading baseline on unseen participants. Treat these forecasts as experimental; the two-hour estimates are particularly uncertain.")
hardware = json.loads((ROOT / "outputs/hardware.json").read_text()) if (ROOT / "outputs/hardware.json").exists() else {}
food_df = foods()
with st.sidebar:
    st.header("Your local profile")
    demo = st.toggle("Demo mode", value=True, help="Demo observations never enter a real profile.")
    name = st.text_input("Profile name", value="My profile")
    st.caption("Stored on this laptop. Profiles separate histories; they are not login accounts. Keep the app on localhost.")
    target = st.number_input("Post-meal reference (mg/dL)", 80, 300, 180,
                             help="Use a target agreed with your care team; this is not a diagnosis threshold.")
    st.markdown("[About the general reference](https://diabetes.org/living-with-diabetes/treatment-care/checking-your-blood-sugar)")
    st.write(f"Population model: **{bundle['name']}**")
    st.caption(f"Version {bundle['version']}")
try:
    profile = profile_key(name, demo)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
rows = history(profile, version=bundle["version"])
tabs = st.tabs(["Forecast & learn", "Foods to review", "Meal history", "Models & validation"])
SAMPLES = {
    "Gentle rise": ([105., 115., 130., 128.], [125., 110.]),
    "Larger rise": ([130., 160., 205., 215.], [225., 195.]),
    "Slower recovery": ([115., 140., 175., 185.], [205., 190.]),
}

with tabs[0]:
    st.subheader("1. Choose a meal and enter its early readings")
    st.caption("Timing starts at the beginning of the meal. The third reading is at 40 minutes. This forecast becomes available at minute 45.")
    sample = st.selectbox("Synthetic test example", list(SAMPLES))
    if st.button("Load test example", disabled=not demo):
        early_sample, later_sample = SAMPLES[sample]
        for minute, value in zip([0, 15, 40, 45], early_sample):
            st.session_state[f"early_{minute}"] = value
        st.session_state["actual60"], st.session_state["actual120"] = later_sample
        match = food_df[food_df.food_name.str.contains("Boiled rice", case=False, na=False)]
        st.session_state["selected_foods"] = [str(match.iloc[0].food_name if len(match) else food_df.iloc[0].food_name)]
        st.session_state.pop("meal_forecast", None)
    early = [c.number_input(f"{minute} min (mg/dL)", 20., 600., value=110., step=1., key=f"early_{minute}")
             for c, minute in zip(st.columns(4), [0, 15, 40, 45])]
    selected = st.multiselect("Foods eaten", sorted(food_df.food_name.tolist()), key="selected_foods")
    quantities = []
    for food in selected:
        unit = str(food_df.loc[food_df.food_name == food, "servings_unit"].iloc[0])
        quantity = st.number_input(f"{food} — servings ({unit})", .25, 20., 1., .25, key=f"qty-{food}")
        quantities.append((food, quantity))
    totals, details = summarize(food_df, quantities)
    if len(details):
        st.dataframe(details, hide_index=True, width="stretch")
        st.caption("Nutrition flags are project heuristics, not a learned or medical verdict that a food is bad.")
    valid_nutrition = not len(details) or not details["Incomplete nutrition"].any()
    if not valid_nutrition:
        st.warning("Missing nutrient values. Choose foods with complete nutrition to train the personal food model.")
    delta, status = correction(rows, early, totals)
    st.caption(f"{status['meals']} measured meals · {status['method']}")
    if st.button("Forecast next readings", type="primary", disabled=not selected or not valid_nutrition):
        base = bundle["model"].predict(np.array([early]))[0]
        st.session_state["meal_forecast"] = {
            "early": early, "foods": quantities, "totals": totals, "base": base.tolist(),
            "forecast": (base + delta).tolist(), "profile": profile, "version": bundle["version"],
        }
    meal = st.session_state.get("meal_forecast")
    current = meal and meal["profile"] == profile and meal["version"] == bundle["version"] and meal["early"] == early and meal["foods"] == quantities
    if meal and not current:
        st.warning("Inputs or profile changed. Generate a fresh forecast before recording an outcome.")
    if current:
        st.subheader("2. Your forecast")
        a, b = st.columns(2)
        a.metric("60-minute estimate", f"{meal['forecast'][0]:.0f} mg/dL")
        b.metric("120-minute estimate", f"{meal['forecast'][1]:.0f} mg/dL")
        st.caption(f"Population validation absolute-error reference (90th percentile): ±{bundle['error90'][0]:.0f} at 60 min, ±{bundle['error90'][1]:.0f} at 120 min. Not a calibrated personal confidence interval.")
        if max(meal["forecast"]) >= target:
            st.warning("Forecast reaches your reference. Review the meal portion and compare with actual measurements.")
        if min(meal["forecast"]) < 20 or max(meal["forecast"]) > 600:
            st.error("Forecast outside the supported range; do not rely on this estimate.")
        graph = pd.DataFrame({"Minute": TIMEPOINTS, "Measured": early + [np.nan, np.nan],
            "Forecast": [np.nan] * 3 + [early[-1]] + meal["forecast"]}).set_index("Minute")
        st.line_chart(graph)
        devices = [k for k, v in hardware.get("devices", {}).items() if v.get("verified")]
        with st.expander("Compare the GPU-trained neural network"):
            st.caption("Comparison candidate. The main forecast uses the validation-selected model.")
            if devices:
                device = st.selectbox("Neural inference device", devices, index=devices.index("NPU") if "NPU" in devices else 0)
                if st.button("Run neural comparison"):
                    try:
                        predicted = np.asarray(neural_device(device)([np.array([early], dtype=np.float32)])[0])[0]
                        st.write(f"{device}: 60 min = {predicted[0]:.1f}; 120 min = {predicted[1]:.1f} mg/dL")
                    except Exception as exc:
                        st.error(f"Device could not run this comparison: {exc}")
        st.subheader("3. Record what actually happened")
        st.caption("Save measured outcomes to update your twin. Never copy predictions into these fields as training labels.")
        with st.form("actual_outcome"):
            a, b = st.columns(2)
            actual60 = a.number_input("Measured at 60 min", 20., 600., value=125., step=1., key="actual60")
            actual120 = b.number_input("Measured at 120 min", 20., 600., value=110., step=1., key="actual120")
            india = timezone(timedelta(hours=5, minutes=30))
            past = datetime.now(india) - timedelta(hours=3)
            date = st.date_input("Meal date (India time)", value=past.date())
            clock = st.time_input("Meal start time (India time)", value=past.time().replace(second=0, microsecond=0))
            verified = st.checkbox("These are measured outcomes (or synthetic values in Demo mode)")
            submitted = st.form_submit_button("Save outcome & update my twin")
        if submitted:
            when = datetime.combine(date, clock, tzinfo=india)
            if not verified:
                st.error("Confirm the source of these outcomes before saving.")
            elif not demo and when + timedelta(hours=2) > datetime.now(india):
                st.error("Two hours must have elapsed since the meal before saving a two-hour outcome.")
            else:
                try:
                    saved = save_observation(profile, meal, [actual60, actual120], when.isoformat())
                    newrows = history(profile, version=bundle["version"])
                    _, updated = correction(newrows, early, totals)
                    st.success(f"Saved. Twin updated from {len(newrows)} measured meals. {updated['method']}" if saved else "This meal time is already recorded; no duplicate training occurred.")
                    if min(actual60, actual120) < 70:
                        st.warning("A measured value is below 70 mg/dL. Follow your established low-glucose care plan; do not rely on the model.")
                    if max(actual60, actual120) >= target:
                        st.warning("A measured outcome reached your reference; this meal is included in your food review history.")
                except ValueError as exc:
                    st.error(str(exc))
        with st.expander("Legacy cohort classifier (research only)"):
            st.caption("Classifies curves by dataset cohort, not diabetes risk. Different measurement protocols and missing data can inflate performance.")
            if st.button("Analyze completed six-point curve"):
                meta = json.loads((ROOT / "outputs/response_model_metrics.json").read_text())
                classifier = joblib.load(ROOT / "models/d1namo_response_model.joblib")
                prob = classifier.predict_proba(prediction_frame(early + [actual60, actual120], meta["features"]))[0, 1]
                st.write(f"Type 1 cohort score: {prob:.3f}. Not a calibrated clinical probability.")

with tabs[1]:
    st.subheader("Foods to limit or review")
    st.write("Review serving size, carbohydrate load and free sugars alongside your actual response. A food is not universally bad for everyone.")
    st.markdown("[Carbohydrates and glucose — American Diabetes Association](https://diabetes.org/food-nutrition/understanding-carbs/get-to-know-carbs)")
    st.caption("Rules per listed serving: free sugars ≥10 g; carbohydrates ≥45 g; or carbohydrates ≥25 g with fibre <3 g. Project heuristics, not clinical thresholds.")
    search = st.text_input("Search foods")
    subset = food_df[food_df.food_name.str.contains(search, case=False, regex=False)]
    _, catalogue = summarize(food_df, [(n, 1.) for n in subset.food_name])
    if len(catalogue):
        flagged = catalogue[catalogue.Review != "No configured nutrition flag"]
        st.dataframe(flagged.sort_values("sugar", ascending=False), hide_index=True, width="stretch")
    st.subheader("Foods in meals that exceeded your reference")
    observed_foods = []
    for r in history(profile):
        for food, qty in r["foods"]:
            observed_foods.append({"Food": food, "Recorded meals": 1,
                "Meals above reference": int(max(r["observed"]) >= target), "2h glucose": r["observed"][1]})
    if observed_foods:
        table = pd.DataFrame(observed_foods).groupby("Food").agg({"Recorded meals": "sum", "Meals above reference": "sum", "2h glucose": "mean"}).reset_index()
        st.dataframe(table.sort_values("Meals above reference", ascending=False), hide_index=True, width="stretch")
        st.caption("Whole-meal associations, not proof that one ingredient caused the response. Insulin, exercise, sleep and illness also matter.")
    else:
        st.write("Save measured outcomes to build your personal food review list.")

with tabs[2]:
    records = history(profile)
    st.subheader(f"{len(records)} saved meals")
    if records:
        table = pd.DataFrame([{"Meal time": r["eaten_at"], "Foods": ", ".join(f"{n} × {q}" for n, q in r["foods"]),
            "Forecast 60": r["forecast"][0], "Measured 60": r["observed"][0],
            "Forecast 120": r["forecast"][1], "Measured 120": r["observed"][1], "Model version": r["version"]} for r in records])
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Download this profile's meal history", table.to_csv(index=False), "meal_history.csv", "text/csv")
        st.caption("Historical forecasts are preserved. New population versions start a new personal correction; older history remains available.")

with tabs[3]:
    st.subheader("Training and validation")
    st.write(f"{report['n_meals']} complete curves from {report['n_subjects']} participants. Selected model: **{report['selected']}**.")
    st.dataframe(pd.DataFrame(report["cv"]).T.rename_axis("Candidate"), width="stretch")
    st.write("Untouched participant holdout (mg/dL)")
    st.json({"selected_model": report["holdout"], "persistence_baseline": report["holdout_persistence"]})
    st.caption(report["note"])
    st.write(f"Neural-network training: {report['gpu_training']}")
    st.json(hardware)
    st.write("Personal learning starts with a regularized offset. From eight meals, a nutrient-aware ridge candidate must improve later-meal validation error by at least 5%. Corrections are capped at ±40 mg/dL as a prototype guardrail, not a safety guarantee.")

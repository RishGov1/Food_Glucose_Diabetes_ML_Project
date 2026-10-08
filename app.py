from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

sys.path.append(str(Path(__file__).resolve().parent))
from src.features import six_point_summary

st.set_page_config(
    page_title="Food + Glucose Diabetes ML",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️ Food + Post-Meal Glucose Diabetes Risk")

model_path = Path("models/d1namo_glucose_only_model.joblib")
if not model_path.exists():
    st.warning("Train the project first with: python run_all.py")
    st.stop()

model = joblib.load(model_path)

st.subheader("Six-point post-meal glucose input")
c = st.columns(6)
labels = [
    ("Immediately after", "g0"),
    ("15 min", "g15"),
    ("40 min", "g40"),
    ("45 min", "g45"),
    ("1 hour", "g60"),
    ("2 hours", "g120"),
]

vals = []
for col, (label, key) in zip(c, labels):
    with col:
        vals.append(
            st.number_input(
                label,
                min_value=20.0,
                max_value=600.0,
                value=120.0,
                step=1.0,
                key=key,
            )
        )

st.subheader("Optional food intake")
f1, f2, f3, f4 = st.columns(4)
with f1:
    carbs = st.number_input("Carbohydrates (g)", min_value=0.0, value=50.0)
with f2:
    sugar = st.number_input("Sugar (g)", min_value=0.0, value=10.0)
with f3:
    protein = st.number_input("Protein (g)", min_value=0.0, value=20.0)
with f4:
    fat = st.number_input("Fat (g)", min_value=0.0, value=15.0)

if st.button("Analyze glucose response", type="primary"):
    summary = six_point_summary(vals)

    X = pd.DataFrame({"g_120": [vals[-1]]})
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)

    st.subheader("Model output")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Estimated diabetes probability", f"{prob*100:.1f}%")
    with m2:
        st.metric("2-hour glucose", f"{vals[-1]:.1f} mg/dL")
    with m3:
        st.metric("Peak glucose", f"{summary['peak_glucose']:.1f} mg/dL")

    if pred == 1:
        st.error("Model classification: higher diabetes likelihood")
    else:
        st.success("Model classification: lower diabetes likelihood")

    st.subheader("Post-meal response analysis")
    a = st.columns(4)
    metrics = [
        ("Peak time", f"{summary['peak_time_min']:.0f} min"),
        ("Peak excursion", f"{summary['peak_excursion']:.1f} mg/dL"),
        ("2h excursion", f"{summary['two_hour_excursion']:.1f} mg/dL"),
        ("AUC (0–120)", f"{summary['auc_0_120']:.0f} mg·min/dL"),
        ("0–15 slope", f"{summary['slope_0_15']:.2f} mg/dL/min"),
        ("60–120 slope", f"{summary['slope_60_120']:.2f} mg/dL/min"),
        ("60–120 recovery", f"{summary['recovery_60_120']:.1f} mg/dL"),
        ("Peak rise", f"{summary['pct_peak_rise']:.1f}%"),
    ]
    for i, (name, value) in enumerate(metrics):
        with a[i % 4]:
            st.metric(name, value)

    chart = pd.DataFrame({
        "Minutes": [0, 15, 40, 45, 60, 120],
        "Glucose": vals,
    }).set_index("Minutes")
    st.line_chart(chart)

    st.subheader("Food intake recorded")
    st.write({
        "Carbohydrates (g)": carbs,
        "Sugar (g)": sugar,
        "Protein (g)": protein,
        "Fat (g)": fat,
    })

    st.warning(
        "This prototype is for education/research. A probability from this model "
        "must not be interpreted as a clinical diagnosis."
    )

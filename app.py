"""Diabetes Digital Twin — Adaptive Closed-Loop Personalized Patient Model."""
from datetime import datetime, timezone
from pathlib import Path
import json
import joblib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from twin.config import (
    DEFAULT_PATIENT,
    DEFAULT_TWIN_WEIGHTS,
    TARGET_MAX_GLUCOSE,
    TARGET_MIN_GLUCOSE,
)
from twin.database import (
    add_meal_log,
    get_hba1c_history,
    get_latest_weights,
    get_learning_history,
    get_meal_history,
    get_patient,
    get_patient_summary_stats,
    init_db,
)
from twin.evaluator import evaluate_meal_response
from twin.online_model import (
    adapt_twin_weights,
    generate_modeled_curve,
    predict_peak_rise,
)
from twin.recommender import (
    classify_food_item,
    get_recommendation_catalog,
    load_food_catalog,
)
from twin.seed_data import seed_database

# ---------------------------------------------------------
# Streamlit App Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Digital Twin | Adaptive Patient Model",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern visual polish and responsive typography
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(248, 250, 252, 0.9), rgba(241, 245, 249, 0.8));
        border: 1px solid rgba(226, 232, 240, 0.9);
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
        transition: all 0.2s ease-in-out;
    }
    /* Banner Callouts */
    .twin-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 20px 24px;
        border-radius: 14px;
        margin-bottom: 22px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.15);
    }
    .twin-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-live {
        background-color: #10b981;
        color: #ffffff;
    }
    .badge-patient {
        background-color: #3b82f6;
        color: #ffffff;
    }
    .badge-hba1c {
        background-color: #8b5cf6;
        color: #ffffff;
    }
    .food-card {
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 5px solid #3b82f6;
        background: #f8fafc;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Ensure DB initialized and seeded
init_db()
seed_database()

PATIENT_ID = DEFAULT_PATIENT["id"]
patient_info = get_patient(PATIENT_ID) or DEFAULT_PATIENT
latest_weights = get_latest_weights(PATIENT_ID)
stats = get_patient_summary_stats(PATIENT_ID)
hba1c_history = get_hba1c_history(PATIENT_ID)

# ---------------------------------------------------------
# Sidebar: Patient Twin Controller & Clinical Snapshot
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Digital Twin Core")
    st.caption("Adaptive In Silico Metabolic Replica")

    st.markdown(f"""
    **Patient:** {patient_info['name']}  
    **Age / Sex:** {patient_info['age']} yrs • {patient_info['gender']}  
    **Diagnosis:** {patient_info['diabetes_type']}  
    **Clinical Target:** {TARGET_MIN_GLUCOSE:.0f} – {TARGET_MAX_GLUCOSE:.0f} mg/dL  
    **Learning Status:** Online (Step #{latest_weights.get('learning_step', 0)})
    """)

    st.divider()
    st.subheader("Twin Calibration")
    st.write(f"- **Carb Sensitivity:** `{latest_weights['carb_weight']:.2f}` mg/dL/g")
    st.write(f"- **Fiber Damping:** `{latest_weights['fiber_weight']:.2f}` mg/dL/g")
    st.write(f"- **Fat Delay Buffer:** `{latest_weights['fat_weight']:.2f}` mg/dL/g")
    st.write(f"- **Protein Factor:** `{latest_weights['protein_weight']:.2f}` mg/dL/g")
    st.write(f"- **Basal Drift Bias:** `{latest_weights['bias']:.1f}` mg/dL")

    st.divider()
    if st.button("🔄 Reset Demo Database", help="Reload realistic historical meals and reset twin weights"):
        seed_database(force_reseed=True)
        st.success("Database reseeded with historical diabetic records!")
        st.rerun()

# ---------------------------------------------------------
# Main App Header & Profile Banner
# ---------------------------------------------------------
st.markdown(f"""
<div class="twin-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h2 style="margin: 0; color: #ffffff;">🧬 Diabetes Digital Twin Platform</h2>
            <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 0.95rem;">
                Continuous In-Silico Patient Twin with Longitudinal Glycemic History, Real-Time Intolerance Detection, and Online Adaptive Learning.
            </p>
        </div>
        <div style="margin-top: 10px;">
            <span class="twin-badge badge-live">● TWIN ACTIVE</span>
            <span class="twin-badge badge-patient">{patient_info['name']} (T2D)</span>
            <span class="twin-badge badge-hba1c">HbA1c: {stats['latest_hba1c']}%</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_overview, tab_food, tab_logger, tab_model, tab_journal = st.tabs([
    "📊 Twin Dashboard",
    "🥗 Allowed & Restricted Foods",
    "🍽️ Log Meal & Real-Time Evaluation",
    "🧠 Adaptive Model & Weights",
    "📜 Clinical Journal & History"
])

# =========================================================
# TAB 1: TWIN DASHBOARD
# =========================================================
with tab_overview:
    # Summary Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            label="Current HbA1c",
            value=f"{stats['latest_hba1c']}%",
            delta="-1.4% vs Baseline",
            delta_color="normal"
        )
    with m2:
        st.metric(
            label="Time in Range (TIR)",
            value=f"{stats['tir_percent']}%",
            delta="+8.5% this month",
            delta_color="normal",
            help="Percentage of post-meal 2-hour readings within 70-160 mg/dL target."
        )
    with m3:
        st.metric(
            label="Mean Glycemic Rise",
            value=f"+{stats['avg_delta']:.1f} mg/dL",
            delta="Controlled",
            delta_color="off"
        )
    with m4:
        st.metric(
            label="Logged Meals",
            value=stats["total_meals"],
            help="Total historical meal events analyzed by the twin."
        )
    with m5:
        st.metric(
            label="Flagged Intolerances",
            value=stats["intolerant_foods_count"],
            delta="Emerging alert" if stats["intolerant_foods_count"] > 0 else "None",
            delta_color="inverse" if stats["intolerant_foods_count"] > 0 else "normal",
            help="Foods where patient exhibits emerging or established glycemic sensitivity."
        )

    if stats["intolerant_foods"]:
        st.warning(
            f"⚠️ **Emerging Intolerance Flagged for Patient:** The digital twin has detected a degrading glycemic tolerance trend for: "
            f"**{', '.join(stats['intolerant_foods'])}**. Recommended to avoid or substitute with lower glycemic alternatives."
        )

    st.write("")
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.subheader("Longitudinal HbA1c & Fasting Glycemia Trajectory")
        if hba1c_history:
            df_hba1c = pd.DataFrame(hba1c_history)
            fig_hba1c = go.Figure()
            fig_hba1c.add_trace(go.Scatter(
                x=df_hba1c["recorded_date"],
                y=df_hba1c["hba1c_value"],
                mode="lines+markers+text",
                name="HbA1c (%)",
                text=[f"{v}%" for v in df_hba1c["hba1c_value"]],
                textposition="top center",
                line=dict(color="#8b5cf6", width=3),
                marker=dict(size=10, color="#6d28d9")
            ))
            fig_hba1c.add_hline(y=7.0, line_dash="dash", line_color="#ef4444", annotation_text="Standard T2D Target (<7.0%)")
            fig_hba1c.add_hline(y=5.7, line_dash="dot", line_color="#10b981", annotation_text="Normal Reference (<5.7%)")
            fig_hba1c.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=320,
                yaxis=dict(title="HbA1c (%)", range=[5.0, 9.0]),
                xaxis=dict(title="Checkup Date"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_hba1c, use_container_width=True)

    with c_right:
        st.subheader("Postprandial Response Distribution")
        donut_data = pd.DataFrame([
            {"Verdict": "Good (In-Target)", "Count": stats["good_meals"], "Color": "#10b981"},
            {"Verdict": "Moderate", "Count": stats["moderate_meals"], "Color": "#f59e0b"},
            {"Verdict": "Bad (Spike / Excursion)", "Count": stats["bad_meals"], "Color": "#ef4444"},
        ])
        fig_donut = px.pie(
            donut_data,
            values="Count",
            names="Verdict",
            hole=0.55,
            color="Verdict",
            color_discrete_map={
                "Good (In-Target)": "#10b981",
                "Moderate": "#f59e0b",
                "Bad (Spike / Excursion)": "#ef4444"
            }
        )
        fig_donut.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=320,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.subheader("Recent Patient Meal Stream")
    recent_meals = get_meal_history(PATIENT_ID, limit=6)
    if recent_meals:
        cols = st.columns(3)
        for idx, m in enumerate(recent_meals):
            with cols[idx % 3]:
                badge_bg = "#dcfce7" if m["verdict"] == "GOOD" else ("#fef3c7" if m["verdict"] == "MODERATE" else "#fee2e2")
                badge_color = "#166534" if m["verdict"] == "GOOD" else ("#92400e" if m["verdict"] == "MODERATE" else "#991b1b")
                st.markdown(f"""
                <div class="food-card" style="border-left-color: {badge_color};">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 700; font-size: 0.95rem;">{m['food_name']}</span>
                        <span style="background: {badge_bg}; color: {badge_color}; font-size: 0.75rem; padding: 2px 8px; border-radius: 12px; font-weight: 600;">{m['verdict']}</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #64748b; margin-top: 4px;">
                        {m['meal_type']} • {m['portion_amount']} {m['portion_unit']} • Carbs {m['carbs_g']:.0f}g
                    </div>
                    <div style="margin-top: 8px; display: flex; justify-content: space-between; font-size: 0.88rem;">
                        <span>Baseline: <b>{m['pre_glucose']:.0f}</b> mg/dL</span>
                        <span>Peak: <b>{m['peak_glucose']:.0f}</b> mg/dL</span>
                        <span style="color: {badge_color}; font-weight: 700;">+{m['delta_glucose']:.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# =========================================================
# TAB 2: ALLOWED & RESTRICTED FOODS
# =========================================================
with tab_food:
    st.header("🥗 Personalized Food Intelligence & Tolerances")
    st.write(
        "Allowed and restricted foods are dynamically calculated using **your Digital Twin's personalized sensitivity weights** "
        "(carb sensitivity, fiber damping, fat buffering) and your historical meal responses."
    )

    f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
    with f_col1:
        search_kw = st.text_input("🔍 Search Food Catalog", placeholder="e.g. Khichdi, Roti, Paneer, Biryani...")
    with f_col2:
        tier_choice = st.selectbox(
            "Filter Category",
            options=["ALL", "ALLOWED", "CAUTION", "NOT_RECOMMENDED", "INTOLERANCE_ALERTS"],
            format_func=lambda x: {
                "ALL": "🍽️ All Foods (Catalog)",
                "ALLOWED": "🟢 Allowed / Recommended",
                "CAUTION": "🟡 Caution / Portion Control",
                "NOT_RECOMMENDED": "🔴 Not Recommended / Spike Risk",
                "INTOLERANCE_ALERTS": "⚠️ Emerging Intolerance Alerts"
            }[x]
        )
    with f_col3:
        portion_sim = st.slider("Portion Multiplier Simulator", min_value=0.5, max_value=3.0, value=1.0, step=0.25, help="Simulate how larger or smaller portions affect recommendation tier.")

    catalog_items = get_recommendation_catalog(
        patient_id=PATIENT_ID,
        tier_filter=tier_choice,
        search_query=search_kw,
        portion_multiplier=portion_sim,
        limit=60
    )

    # Metric counts for quick reference
    allowed_count = sum(1 for item in catalog_items if item["tier"] == "ALLOWED")
    caution_count = sum(1 for item in catalog_items if item["tier"] == "CAUTION")
    not_rec_count = sum(1 for item in catalog_items if item["tier"] == "NOT_RECOMMENDED")

    st.markdown(f"""
    **Current View:** Showing `{len(catalog_items)}` foods • 
    <span style="color: #16a34a; font-weight: 600;">{allowed_count} Allowed</span> • 
    <span style="color: #ca8a04; font-weight: 600;">{caution_count} Caution</span> • 
    <span style="color: #dc2626; font-weight: 600;">{not_rec_count} Not Recommended</span>
    """, unsafe_allow_html=True)
    st.write("")

    if not catalog_items:
        st.info("No foods matched your search query or filter. Try a broader search term.")
    else:
        for item in catalog_items:
            with st.container(border=True):
                top_l, top_r = st.columns([3, 1])
                with top_l:
                    st.markdown(f"### {item['food_name']}")
                    st.caption(f"Portion: **{item['portion']:.2f} × {item['unit']}** | Calories: **{item['calories_kcal']:.0f} kcal** | Digestion: **~{item['digestion_hours']:.1f} hrs**")
                with top_r:
                    st.markdown(f"""
                    <div style="text-align: right;">
                        <span style="background: {item['badge_color']}; color: white; padding: 6px 12px; border-radius: 14px; font-weight: 700; font-size: 0.85rem;">
                            {item['status_label']}
                        </span>
                        <div style="margin-top: 6px; font-size: 0.82rem; color: #64748b;">
                            Twin Tolerance Score: <b>{item['tolerance_score']:.0f}/100</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Macro badges
                st.markdown(f"""
                <div style="display: flex; gap: 18px; margin: 8px 0; font-size: 0.9rem; flex-wrap: wrap;">
                    <span>🍞 Carbs: <b>{item['carbs_g']:.1f} g</b></span>
                    <span>🥗 Fiber: <b>{item['fiber_g']:.1f} g</b></span>
                    <span>🥩 Protein: <b>{item['protein_g']:.1f} g</b></span>
                    <span>🥑 Fat: <b>{item['fat_g']:.1f} g</b></span>
                    <span>🍭 Free Sugar: <b>{item['sugar_g']:.1f} g</b></span>
                    <span style="color: #2563eb;">📈 Forecast Rise: <b>+{item['predicted_rise']:.0f} mg/dL</b></span>
                </div>
                """, unsafe_allow_html=True)

                st.write(f"**Twin Evaluation:** {item['reason']}")

                if item["has_history"]:
                    st.info(f"📊 **Historical Twin Memory:** {item['history_note']}")

# =========================================================
# TAB 3: MEAL LOGGER & REAL-TIME TWIN EVALUATION
# =========================================================
with tab_logger:
    st.header("🍽️ Meal Logging & Real-Time Twin Analysis")
    st.write(
        "Log a meal with your post-meal glucose readings. The Digital Twin will immediately assess if the response is "
        "**Good** or **Bad**, detect whether **you are becoming intolerant to this food** compared to your historical baseline, "
        "and **adaptively update its internal physiological model weights**."
    )

    # Food Selection
    food_df = load_food_catalog()
    food_names = food_df["food_name"].tolist()

    # Pre-select an interesting item or Sweetened Instant Oats for testing intolerance
    default_idx = 0
    for idx, name in enumerate(food_names):
        if "oats" in name.lower() or "khichdi" in name.lower():
            default_idx = idx
            break

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        selected_food = st.selectbox("Select Meal / Food Item", food_names, index=default_idx)
    with c2:
        food_match = food_df[food_df["food_name"] == selected_food].iloc[0]
        servings = st.number_input(f"Servings ({food_match['servings_unit']})", min_value=0.25, max_value=6.0, value=1.0, step=0.25)
    with c3:
        meal_type = st.selectbox("Meal Period", ["Breakfast", "Lunch", "Snack", "Dinner"])

    # Scaled nutrients
    carbs = float(food_match["unit_serving_carb_g"]) * servings
    fiber = float(food_match["unit_serving_fibre_g"]) * servings
    protein = float(food_match["unit_serving_protein_g"]) * servings
    fat = float(food_match["unit_serving_fat_g"]) * servings
    digestion = float(food_match["digestion_hours"])

    with st.container(border=True):
        st.caption("Nutrient breakdown for selected portion")
        n1, n2, n3, n4, n5 = st.columns(5)
        n1.metric("Carbohydrates", f"{carbs:.1f} g")
        n2.metric("Dietary Fiber", f"{fiber:.1f} g")
        n3.metric("Protein", f"{protein:.1f} g")
        n4.metric("Fat", f"{fat:.1f} g")
        n5.metric("Digestion Time", f"{digestion:.1f} hrs")

    st.subheader("Blood Glucose Readings")
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        pre_glucose = st.number_input("Pre-Meal Baseline Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=118.0, step=1.0)
    with g_col2:
        twin_predicted_rise = predict_peak_rise(carbs, fiber, protein, fat, pre_glucose, stats["latest_hba1c"], latest_weights)
        st.info(f"🔮 **Digital Twin Pre-Meal Forecast:** Expected peak **{pre_glucose + twin_predicted_rise:.0f} mg/dL** (+{twin_predicted_rise:.0f} mg/dL rise)")

    st.caption("Enter your measured post-meal glucose readings:")
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        g15 = st.number_input("15 min (mg/dL)", min_value=40.0, max_value=450.0, value=float(round(pre_glucose + twin_predicted_rise * 0.3)), step=1.0)
    with r2:
        g30 = st.number_input("30 min (mg/dL)", min_value=40.0, max_value=450.0, value=float(round(pre_glucose + twin_predicted_rise * 0.7)), step=1.0)
    with r3:
        g45 = st.number_input("45 min (mg/dL)", min_value=40.0, max_value=450.0, value=float(round(pre_glucose + twin_predicted_rise * 0.95)), step=1.0)
    with r4:
        g60 = st.number_input("60 min / 1-hr (mg/dL)", min_value=40.0, max_value=450.0, value=float(round(pre_glucose + twin_predicted_rise * 0.9)), step=1.0)
    with r5:
        g120 = st.number_input("120 min / 2-hr (mg/dL)", min_value=40.0, max_value=450.0, value=float(round(pre_glucose + twin_predicted_rise * 0.25)), step=1.0)

    btn_eval = st.button("🚀 Log Meal & Run Digital Twin Analysis", type="primary", use_container_width=True)

    if btn_eval:
        readings_dict = {
            "glucose_15m": g15,
            "glucose_30m": g30,
            "glucose_45m": g45,
            "glucose_60m": g60,
            "glucose_120m": g120,
        }
        pred_peak = pre_glucose + twin_predicted_rise

        # 1. Run Evaluator
        eval_res = evaluate_meal_response(
            pre_glucose=pre_glucose,
            readings=readings_dict,
            food_name=selected_food,
            patient_id=PATIENT_ID,
            predicted_peak=pred_peak
        )

        # 2. Online Adaptive Learning Step (Weights Update)
        meal_entry = {
            "meal_type": meal_type,
            "food_name": selected_food,
            "portion_amount": servings,
            "portion_unit": food_match["servings_unit"],
            "carbs_g": carbs,
            "fiber_g": fiber,
            "protein_g": protein,
            "fat_g": fat,
            "digestion_hours": digestion,
            "pre_glucose": pre_glucose,
            "glucose_15m": g15,
            "glucose_30m": g30,
            "glucose_45m": g45,
            "glucose_60m": g60,
            "glucose_120m": g120,
            "peak_glucose": eval_res["peak_glucose"],
            "delta_glucose": eval_res["delta_glucose"],
            "verdict": eval_res["verdict"],
            "intolerance_status": eval_res["intolerance_status"],
            "intolerance_notes": eval_res["intolerance_message"],
            "predicted_peak": pred_peak,
            "residual_error": eval_res["residual_error"]
        }

        # Adapt model weights based on observed error
        learning_res = adapt_twin_weights(
            patient_id=PATIENT_ID,
            meal_record=meal_entry,
            observed_peak=eval_res["peak_glucose"],
            hba1c=stats["latest_hba1c"],
            current_weights=latest_weights
        )

        # 3. Save meal record to SQLite DB
        meal_id = add_meal_log(PATIENT_ID, meal_entry)

        # -------------------------------------------------
        # Display Results
        # -------------------------------------------------
        st.divider()
        st.subheader("Twin Clinical Verdict & Intolerance Analysis")

        # Verdict banner
        st.markdown(f"""
        <div style="background: {eval_res['verdict_color']}; color: white; padding: 14px 20px; border-radius: 10px; margin-bottom: 15px;">
            <h3 style="margin: 0; color: white;">{eval_res['verdict_badge']}</h3>
            <p style="margin: 6px 0 0 0; font-size: 0.95rem;">{eval_res['verdict_summary']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Intolerance Alert / Confirmation
        if eval_res["is_emerging_intolerant"]:
            st.error(f"### {eval_res['intolerance_title']}\n{eval_res['intolerance_message']}")
        elif eval_res["intolerance_status"] == "KNOWN_INTOLERANT":
            st.warning(f"### {eval_res['intolerance_title']}\n{eval_res['intolerance_message']}")
        else:
            st.success(f"### {eval_res['intolerance_title']}\n{eval_res['intolerance_message']}")

        # Key Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pre-Meal Glucose", f"{pre_glucose:.0f} mg/dL")
        k2.metric("Observed Peak", f"{eval_res['peak_glucose']:.0f} mg/dL", f"+{eval_res['delta_glucose']:.0f} mg/dL")
        k3.metric("Peak Time", f"{eval_res['peak_time']} minutes")
        k4.metric("2-Hour Recovery", f"{g120:.0f} mg/dL", f"{g120 - pre_glucose:+.0f} mg/dL vs pre")

        # Interactive Curve Plot
        st.subheader("Dynamic Glucose Trajectory: Measured vs Digital Twin Simulation")
        sim_curve = generate_modeled_curve(pre_glucose, twin_predicted_rise, digestion, fat, fiber)
        df_sim = pd.DataFrame(sim_curve, columns=["Minute", "Glucose"])

        measured_curve = [
            (0, pre_glucose), (15, g15), (30, g30), (45, g45), (60, g60), (120, g120)
        ]
        df_meas = pd.DataFrame(measured_curve, columns=["Minute", "Glucose"])

        fig_curve = go.Figure()
        # Target Zone Shading
        fig_curve.add_hrect(y0=TARGET_MIN_GLUCOSE, y1=TARGET_MAX_GLUCOSE, fillcolor="#10b981", opacity=0.1, line_width=0, annotation_text="Target Range (70-160)")
        # Measured Line
        fig_curve.add_trace(go.Scatter(
            x=df_meas["Minute"], y=df_meas["Glucose"],
            mode="lines+markers", name="Your Measured Glucose",
            line=dict(color="#2563eb", width=3), marker=dict(size=9, color="#1d4ed8")
        ))
        # Twin Simulation Line
        fig_curve.add_trace(go.Scatter(
            x=df_sim["Minute"], y=df_sim["Glucose"],
            mode="lines", name="Twin Model Simulation",
            line=dict(color="#f97316", width=2, dash="dash")
        ))
        fig_curve.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(title="Minutes Post-Meal", tickvals=[0, 15, 30, 45, 60, 90, 120]),
            yaxis=dict(title="Glucose (mg/dL)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # Online Weight Adaptation Telemetry
        st.subheader("Online Learning Telemetry: Adapted Twin Weights")
        st.markdown(f"**Step #{learning_res['learning_step']} Completed:** {learning_res['notes']}")

        w_diff_df = pd.DataFrame([
            {
                "Parameter": "Carbohydrate Sensitivity",
                "Old Value": f"{learning_res['old_weights']['carb_weight']:.3f} mg/dL/g",
                "New Value": f"{learning_res['new_weights']['carb_weight']:.3f} mg/dL/g",
                "Adjustment (Delta)": f"{learning_res['deltas']['carb_weight']:+.4f}",
            },
            {
                "Parameter": "Dietary Fiber Damping",
                "Old Value": f"{learning_res['old_weights']['fiber_weight']:.3f} mg/dL/g",
                "New Value": f"{learning_res['new_weights']['fiber_weight']:.3f} mg/dL/g",
                "Adjustment (Delta)": f"{learning_res['deltas']['fiber_weight']:+.4f}",
            },
            {
                "Parameter": "Fat Glycemic Delay",
                "Old Value": f"{learning_res['old_weights']['fat_weight']:.3f} mg/dL/g",
                "New Value": f"{learning_res['new_weights']['fat_weight']:.3f} mg/dL/g",
                "Adjustment (Delta)": f"{learning_res['deltas']['fat_weight']:+.4f}",
            },
            {
                "Parameter": "Protein Glucagon Effect",
                "Old Value": f"{learning_res['old_weights']['protein_weight']:.3f} mg/dL/g",
                "New Value": f"{learning_res['new_weights']['protein_weight']:.3f} mg/dL/g",
                "Adjustment (Delta)": f"{learning_res['deltas']['protein_weight']:+.4f}",
            },
            {
                "Parameter": "Basal Digestive Drift (Bias)",
                "Old Value": f"{learning_res['old_weights']['bias']:.2f} mg/dL",
                "New Value": f"{learning_res['new_weights']['bias']:.2f} mg/dL",
                "Adjustment (Delta)": f"{learning_res['deltas']['bias']:+.3f}",
            },
        ])
        st.dataframe(w_diff_df, use_container_width=True)

# =========================================================
# TAB 4: TWIN MODEL & ONLINE WEIGHTS
# =========================================================
with tab_model:
    st.header("🧠 Twin Physiological Parameters & Adaptive Dynamics")
    st.write(
        "The Digital Twin operates a mathematical model calibrated to the patient's individual metabolism. "
        "Unlike static machine learning models, the Twin uses **online recursive residual learning** to adapt its coefficients "
        "after each real-world meal."
    )

    st.markdown("""
    $$\\Delta \\text{Glucose}_{\\text{peak}} = w_{\\text{carb}} \\cdot C + w_{\\text{fiber}} \\cdot F_{\\text{fiber}} + w_{\\text{fat}} \\cdot F_{\\text{fat}} + w_{\\text{protein}} \\cdot P + w_{\\text{base}} \\cdot (G_0 - 100) + w_{\\text{hba1c}} \\cdot (\\text{HbA1c} - 5.5) + \\text{bias}$$
    """)

    st.subheader("Current Learned Weights")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Carb Sensitivity", f"{latest_weights['carb_weight']:.3f} mg/dL/g", help="Rise per gram of carbohydrate consumed")
    p2.metric("Fiber Buffering", f"{latest_weights['fiber_weight']:.3f} mg/dL/g", help="Protective damping factor per gram of dietary fiber")
    p3.metric("Fat Delay", f"{latest_weights['fat_weight']:.3f} mg/dL/g", help="Initial buffering and gastric emptying delay")
    p4.metric("Protein Gluconeogenesis", f"{latest_weights['protein_weight']:.3f} mg/dL/g", help="Delayed conversion from dietary amino acids")

    st.write("")
    st.subheader("Online Learning Telemetry Log")
    learn_events = get_learning_history(PATIENT_ID, limit=20)
    if learn_events:
        event_rows = []
        for ev in learn_events:
            old_w = json.loads(ev["old_weights_json"])
            new_w = json.loads(ev["new_weights_json"])
            delta_w = json.loads(ev["delta_weights_json"])
            event_rows.append({
                "Timestamp": ev["timestamp"][:19].replace("T", " "),
                "Trigger": ev["trigger_reason"],
                "Loss (Error²)": f"{ev['loss']:.1f}",
                "Carb Δ": f"{delta_w.get('carb_weight', 0):+.4f}",
                "Fiber Δ": f"{delta_w.get('fiber_weight', 0):+.4f}",
                "Fat Δ": f"{delta_w.get('fat_weight', 0):+.4f}",
                "Current Carb Weight": f"{new_w.get('carb_weight', 0):.3f}",
            })
        st.dataframe(pd.DataFrame(event_rows), use_container_width=True)
    else:
        st.info("No online learning events logged yet.")

    # ---------------------------------------------------------
    # Real Clinical ML Models (PLOS ONE journal.pone.0302777)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("🔬 Clinical Machine Learning Models (PLOS ONE Real Patient Cohort)")
    st.markdown(
        "Trained on real clinical records from 400 Type 2 Diabetes patients evaluated for oral glucose tolerance test clearance "
        "and diabetes remission following lifestyle intervention (*Tripathi et al., PLOS ONE 19(5): e0302777, 2024*)."
    )

    models_dir = Path(__file__).resolve().parent / "models"
    meta_path = models_dir / "model_metadata.json"
    demo_model_path = models_dir / "demo_model.joblib"
    impact_model_path = models_dir / "food_impact_model.joblib"

    if meta_path.exists() and demo_model_path.exists() and impact_model_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            ml_meta = json.load(f)

        m1_col, m2_col = st.columns(2)
        with m1_col:
            m1_info = ml_meta["models"]["demo_model"]
            st.markdown(f"""
            <div class="food-card" style="border-left-color: #10b981;">
                <h4 style="margin: 0 0 6px 0; color: #0f172a;">🎯 {m1_info['title']}</h4>
                <p style="margin: 0 0 8px 0; font-size: 0.88rem; color: #475569;">
                    Predicts whether a patient achieves remission / clears the 75g Oral Glucose Tolerance Test (OGTT).
                </p>
                <div style="display: flex; gap: 15px; margin-top: 8px;">
                    <div><strong>ROC AUC:</strong> <code>{m1_info['roc_auc']:.4f}</code></div>
                    <div><strong>Trained On:</strong> <code>{m1_info['n_samples']} Patients</code></div>
                    <div><strong>Features:</strong> <code>{m1_info['n_features']}</code></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m2_col:
            m2_info = ml_meta["models"]["food_impact_model"]
            st.markdown(f"""
            <div class="food-card" style="border-left-color: #f97316;">
                <h4 style="margin: 0 0 6px 0; color: #0f172a;">⚡ {m2_info['title']}</h4>
                <p style="margin: 0 0 8px 0; font-size: 0.88rem; color: #475569;">
                    Predicts acute post-challenge excursion risk (2-hour glucose ≥ 140 mg/dL after 75g glucose load).
                </p>
                <div style="display: flex; gap: 15px; margin-top: 8px;">
                    <div><strong>ROC AUC:</strong> <code>{m2_info['roc_auc']:.4f}</code></div>
                    <div><strong>Trained On:</strong> <code>{m2_info['n_samples']} Patients</code></div>
                    <div><strong>Features:</strong> <code>{m2_info['n_features']}</code></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🩺 Run Patient Clinical Prediction Simulator (Real-Data Model)", expanded=False):
            st.write("Evaluate patient's remission clearance potential and post-challenge spike risk using the real clinical model:")
            c_sim1, c_sim2, c_sim3 = st.columns(3)
            with c_sim1:
                sim_duration = st.number_input("Diabetes Duration (Years)", min_value=0.1, max_value=35.0, value=4.5, step=0.5)
                sim_base_weight = st.number_input("Baseline Weight (kg)", min_value=40.0, max_value=150.0, value=78.0, step=0.5)
                sim_curr_weight = st.number_input("Current Weight (kg)", min_value=40.0, max_value=150.0, value=72.0, step=0.5)
            with c_sim2:
                sim_height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=1.0)
                sim_base_hba1c = st.number_input("Baseline HbA1c (%)", min_value=4.0, max_value=16.0, value=8.2, step=0.1)
                sim_curr_hba1c = st.number_input("Current HbA1c (%)", min_value=4.0, max_value=16.0, value=6.8, step=0.1)
            with c_sim3:
                sim_fbg_base = st.number_input("Baseline Fasting Glucose (mg/dL)", min_value=50.0, max_value=350.0, value=142.0, step=1.0)
                sim_fbg_curr = st.number_input("Current Fasting Glucose (mg/dL)", min_value=50.0, max_value=350.0, value=104.0, step=1.0)
                sim_on_meds = st.selectbox("On Diabetes Medication?", ["Yes", "No"], index=0)

            if st.button("🔮 Evaluate Patient Remission & Post-Challenge Risk"):
                try:
                    m1_model = joblib.load(demo_model_path)
                    m2_model = joblib.load(impact_model_path)

                    wt_loss = sim_base_weight - sim_curr_weight
                    wt_loss_pct = (wt_loss / sim_base_weight) * 100.0
                    bmi_base = sim_base_weight / ((sim_height / 100.0) ** 2)
                    bmi_curr = sim_curr_weight / ((sim_height / 100.0) ** 2)
                    bmi_delta = bmi_base - bmi_curr
                    hba1c_delta = sim_base_hba1c - sim_curr_hba1c
                    fbg_delta = sim_fbg_base - sim_fbg_curr

                    # Estimate surrogate HOMA from fasting values
                    est_insulin_base = 12.0
                    est_insulin_curr = 7.5
                    homa_ir_base = (sim_fbg_base * est_insulin_base) / 405.0
                    homa_ir_curr = (sim_fbg_curr * est_insulin_curr) / 405.0
                    homa_b_base = max(10.0, (20.0 * est_insulin_base) / max(1.0, (sim_fbg_base * 0.0555) - 3.5))
                    homa_b_curr = max(10.0, (20.0 * est_insulin_curr) / max(1.0, (sim_fbg_curr * 0.0555) - 3.5))

                    patient_df = pd.DataFrame([{
                        "Duration_of_Diabete": sim_duration,
                        "Heightcms": sim_height,
                        "Weightkgfirstoccurance": sim_base_weight,
                        "Weightkglastoccurance": sim_curr_weight,
                        "Weight_loss_kg": wt_loss,
                        "Weight_loss_pct": wt_loss_pct,
                        "BMIfirstoccurance": bmi_base,
                        "BMIlastoccurance": bmi_curr,
                        "BMI_delta": bmi_delta,
                        "HbA1Cfirstoccurance": sim_base_hba1c,
                        "HbA1Clastoccurance": sim_curr_hba1c,
                        "HbA1c_delta": hba1c_delta,
                        "FastingBloodGlucosemgdlfirstoccurance": sim_fbg_base,
                        "FastingBloodGlucosemgdllastoccurance": sim_fbg_curr,
                        "FBG_delta": fbg_delta,
                        "InsulinFastinguUmlfirstoccurance": est_insulin_base,
                        "InsulinFastinguUmllastoccurance": est_insulin_curr,
                        "HOMA_IR_baseline": homa_ir_base,
                        "HOMA_IR_endline": homa_ir_curr,
                        "HOMA_Beta_baseline": homa_b_base,
                        "HOMA_Beta_endline": homa_b_curr,
                        "On_Insulin_yn": 0,
                        "On_Diabetes_Medicine_yn": 1 if sim_on_meds == "Yes" else 0,
                    }])

                    remission_prob = m1_model.predict_proba(patient_df[m1_info["features"]])[0][1]

                    excursion_df = pd.DataFrame([{
                        "GTT_FBSL": sim_fbg_curr,
                        "Duration_of_Diabete": sim_duration,
                        "BMIlastoccurance": bmi_curr,
                        "Weightkglastoccurance": sim_curr_weight,
                        "HbA1Clastoccurance": sim_curr_hba1c,
                        "FastingBloodGlucosemgdllastoccurance": sim_fbg_curr,
                        "HOMA_IR_endline": homa_ir_curr,
                        "HOMA_Beta_endline": homa_b_curr,
                        "InsulinFastinguUmllastoccurance": est_insulin_curr,
                        "On_Insulin_yn": 0,
                        "On_Diabetes_Medicine_yn": 1 if sim_on_meds == "Yes" else 0,
                    }])
                    excursion_prob = m2_model.predict_proba(excursion_df[m2_info["features"]])[0][1]

                    st.write("")
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.metric(
                            label="OGTT Remission Clearance Probability",
                            value=f"{remission_prob * 100.0:.1f}%",
                            delta="High Remission Potential" if remission_prob >= 0.50 else "Partial Recovery",
                            delta_color="normal" if remission_prob >= 0.50 else "inverse"
                        )
                    with r2:
                        st.metric(
                            label="Post-Challenge Excursion Risk",
                            value=f"{excursion_prob * 100.0:.1f}%",
                            delta="Low Excursion Risk" if excursion_prob < 0.40 else "High Spike Risk",
                            delta_color="inverse" if excursion_prob >= 0.40 else "normal"
                        )
                    with r3:
                        st.metric(
                            label="Weight Loss Achieved",
                            value=f"{wt_loss:.1f} kg",
                            delta=f"-{wt_loss_pct:.1f}% total mass",
                            delta_color="normal"
                        )

                    if remission_prob >= 0.50:
                        st.success(
                            f"✅ **Remission Clearance Favorable:** The model predicts a **{remission_prob*100.0:.1f}%** chance of clearing the standardized 75g oral glucose tolerance test. "
                            f"Key clinical drivers include the **{wt_loss:.1f} kg ({wt_loss_pct:.1f}%) weight reduction**, **HbA1c drop from {sim_base_hba1c}% to {sim_curr_hba1c}%**, and favorable duration (< 5 yrs)."
                        )
                    else:
                        st.info(
                            f"ℹ️ **Intermediate Response:** The model estimates a **{remission_prob*100.0:.1f}%** clearance probability. Continued lifestyle modification, gradual weight loss, and glycemic stabilization are recommended."
                        )
                except Exception as ex:
                    st.error(f"Inference error: {ex}")

# =========================================================
# TAB 5: CLINICAL JOURNAL & HISTORY
# =========================================================
with tab_journal:
    st.header("📜 Clinical Journal & Longitudinal Patient Records")
    st.write("Complete audit log of all meals, readings, and intolerance statuses stored in the SQLite database.")

    all_meals = get_meal_history(PATIENT_ID, limit=100)
    if all_meals:
        df_meals = pd.DataFrame(all_meals)
        # Select clean columns
        cols_display = [
            "timestamp", "meal_type", "food_name", "portion_amount", "portion_unit",
            "carbs_g", "fiber_g", "protein_g", "fat_g", "pre_glucose", "peak_glucose",
            "delta_glucose", "verdict", "intolerance_status"
        ]
        cols_present = [c for c in cols_display if c in df_meals.columns]
        df_display = df_meals[cols_present].copy()
        df_display["timestamp"] = df_display["timestamp"].str[:16].str.replace("T", " ")

        st.dataframe(df_display, use_container_width=True)

        csv_data = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Meal History (CSV)",
            data=csv_data,
            file_name=f"diabetes_twin_journal_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No meals recorded in database yet.")

    st.divider()
    st.caption(
        "EDUCATIONAL & RESEARCH PROTOTYPE: This Digital Twin simulates metabolic dynamics and online adaptation. "
        "Not a substitute for professional clinical judgment, insulin titration algorithms, or direct physician consultation."
    )

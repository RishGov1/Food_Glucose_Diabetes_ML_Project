# 🧬 Diabetes Digital Twin — Adaptive Closed-Loop Patient Model

An in-silico **Diabetes Digital Twin** that simulates an individual diabetic patient's metabolic dynamics, provides personalized **Allowed vs. Not Recommended** food intelligence, detects **emerging food intolerances** in real time, and **adaptively updates its physiological model weights** with each logged meal.

---

## 🌟 Key Features

### 1. 🗄️ In-Silico Patient Database & Longitudinal Tracking
- **Persistent SQLite Database** (`data/local/diabetes_twin.sqlite3`):
  - **Patient Profiles**: Target range (70–160 mg/dL), diabetes classification, and metabolic parameters.
  - **HbA1c Progression History**: Multi-quarter clinical checkpoints (e.g., 8.2% baseline down to 6.8% active management).
  - **Meal & Postprandial Glucose History**: 30+ realistic meals over 30 days capturing baseline, 15m, 30m, 45m, 60m, and 120m readings, peak excursions, and glycemic verdicts.
  - **Online Learning Telemetry**: Logs each adaptive weight update, loss/error, and trigger rationale.

### 2. 🥗 Personalized Food Intelligence (Allowed & Not Recommended)
- Dynamically classifies over 1,100 dishes from the Indian food dataset using the patient's individual twin sensitivity weights:
  - 🟢 **Allowed / Recommended**: Low glycemic impact, high fiber/protein buffering ratio, historically well tolerated ($\Delta \le 35$ mg/dL).
  - 🟡 **Caution / Portion Control**: Moderate glycemic rise ($35 < \Delta \le 52$ mg/dL), requires portion management or pairing with salads/protein.
  - 🔴 **Not Recommended / Spike Risk**: High rapid glycemic load, high free sugars, triggers acute excursions ($\Delta > 52$ mg/dL).
  - ⚠️ **Emerging Intolerance Alerts**: High-priority flags for foods where the patient's tolerance is actively degrading.
- **Portion Simulator**: Dynamically view how portion size scaling shifts food safety tiers.

### 3. ⏱️ Post-Meal Real-Time Evaluation (Good vs. Bad)
- Clinical glycemic classification of entered readings:
  - 🟢 **GOOD**: Peak $\Delta \le 35$ mg/dL, 2-hr $\le 140$ mg/dL, smooth postprandial recovery.
  - 🟡 **MODERATE**: Peak $\Delta \le 55$ mg/dL, 2-hr $\le 170$ mg/dL, borderline elevation.
  - 🔴 **BAD**: Peak $\Delta > 55$ mg/dL or absolute reading $> 180$ mg/dL, acute excursion.

### 4. ⚠️ Emerging Intolerance Detection
- Compares each new meal reading against the patient's **historical baseline response** for that specific food:
  - If a patient previously tolerated an item (e.g. historical average rise $+24$ mg/dL), but recent meals trigger abnormal spikes ($+52$ mg/dL, $+116\%$ increase), the Twin immediately fires an **Emerging Intolerance Alert**!
  - Alerts the user, downgrades the food tier, and recalibrates the Twin's sensitivity weights.

### 5. 🧠 Online Learning & Adaptive Weight Calibration
- Unlike static machine learning models, the Twin operates an online adaptive learning engine:
  $$\Delta \text{Glucose}_{\text{peak}} = w_{\text{carb}} \cdot C + w_{\text{fiber}} \cdot F_{\text{fiber}} + w_{\text{fat}} \cdot F_{\text{fat}} + w_{\text{protein}} \cdot P + w_{\text{base}} \cdot (G_0 - 100) + w_{\text{hba1c}} \cdot (\text{HbA1c} - 5.5) + \text{bias}$$
- After every logged meal, it computes the residual error:
  $$e = \text{Observed Peak} - \text{Predicted Peak}$$
- Updates the personal sensitivity weights via bounded online gradient descent with physiological constraints:
  - **Carbohydrate Sensitivity Factor** ($w_{\text{carb}}$)
  - **Dietary Fiber Damping Factor** ($w_{\text{fiber}}$)
  - **Fat Delay Factor** ($w_{\text{fat}}$)
  - **Protein Gluconeogenesis Factor** ($w_{\text{protein}}$)
  - **Basal Digestive Drift** ($\text{bias}$)

---

## 🏗️ Project Architecture

```
Food_Glucose_Diabetes_ML_Project/
├── app.py                      # Interactive Streamlit Digital Twin Application
├── twin/                       # Core Digital Twin Python Package
│   ├── __init__.py
│   ├── config.py               # Clinical thresholds, target boundaries, physiological bounds
│   ├── database.py             # SQLite manager, schemas, queries, and telemetry logging
│   ├── evaluator.py            # Real-time post-meal evaluation and Intolerance Detection engine
│   ├── online_model.py         # Predictive model, kinetic trajectory simulation, and online SGD weight updates
│   ├── recommender.py          # Food catalog loader and personalized tolerance tiering
│   └── seed_data.py            # Generates realistic patient history and longitudinal HbA1c trajectory
├── tests/
│   ├── test_diabetes_twin.py   # Unit test suite for DB, recommender, evaluator, intolerance & learning
│   └── test_app_integration.py # End-to-end integration tests
├── data/
│   ├── indian_food.csv         # Comprehensive nutrient and digestion dataset
│   └── local/
│       └── diabetes_twin.sqlite3 # Persistent SQLite database
├── requirements.txt            # Project dependencies (Streamlit, Plotly, Pandas, Scikit-learn, etc.)
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13

### 2. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Digital Twin Application
```powershell
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run the Automated Test Suite
```powershell
pytest tests -v
```

---

## 📋 Digital Twin Verification Example: Emerging Intolerance
1. Navigate to the **"🍽️ Log Meal & Real-Time Evaluation"** tab.
2. Select **`Sweetened Instant Oats with Milk`**.
3. In the patient's historical records, this meal was previously well-tolerated with an average rise of $+24$ to $+26$ mg/dL.
4. Enter pre-meal glucose `120 mg/dL` and post-meal readings peaking at `178 mg/dL` ($+58$ mg/dL rise).
5. Click **"Log Meal & Run Digital Twin Analysis"**.
6. **Result**:
   - Verdict: 🔴 **BAD — Glycemic Spike / Excursion**
   - Alert: ⚠️ **Emerging Intolerance Alert!** ("Developing Intolerance Detected! Historically, your Digital Twin recorded a safe average rise of +25 mg/dL. Today's rise of +58 mg/dL shows developing metabolic sensitivity.")
   - Telemetry: **Carb Sensitivity weight adaptively increments** from `0.88` to `0.92` mg/dL/g, and the food is updated across the recommendation catalog!

---

## ⚖️ Educational & Research Disclaimer
This application is an educational prototype and in-silico simulation demonstrating personalized digital twin architectures and online learning. It does not provide medical diagnoses or replace direct consultation with healthcare professionals.

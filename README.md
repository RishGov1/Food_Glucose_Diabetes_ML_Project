# 🧬 Diabetes Digital Twin — Adaptive Closed-Loop Patient Model & Clinical ML

An in-silico **Diabetes Digital Twin** that simulates an individual diabetic patient's metabolic dynamics, provides personalized **Allowed vs. Not Recommended** food intelligence, detects **emerging food intolerances** in real time, **adaptively updates its physiological model weights** with each logged meal, and integrates **real-world machine learning models** trained on clinical patient data (*PLOS ONE 19(5): e0302777*).

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

### 6. 🔬 Real Clinical Machine Learning Models (PLOS ONE journal.pone.0302777)
- Trained on **400 real Type 2 Diabetes patients** from *Tripathi et al. (2024), PLOS ONE 19(5): e0302777*:
  - **Model 1 (`demo_model.joblib`)**: Oral Glucose Tolerance Test (OGTT) Clearance & Remission Classifier (ROC AUC: **0.8449**, Accuracy: **76%**).
  - **Model 2 (`food_impact_model.joblib`)**: Post-Challenge Glycemic Excursion Risk Classifier (ROC AUC: **0.6833**).
  - **In-App Simulator**: Interactive simulator integrated into Tab 4 of the Streamlit application.

---

## 🏗️ Project Architecture & Directory Layout

```
Food_Glucose_Diabetes_ML_Project/
├── app.py                             # Interactive Streamlit Digital Twin Application
├── train_model.py                     # ML training pipeline for real clinical dataset (PLOS ONE)
├── twin/                              # Core Digital Twin Python Package
│   ├── __init__.py
│   ├── config.py                      # Clinical thresholds, target boundaries, physiological bounds
│   ├── database.py                    # SQLite manager, schemas, queries, and telemetry logging
│   ├── evaluator.py                   # Real-time post-meal evaluation and Intolerance Detection engine
│   ├── online_model.py                # Predictive model, kinetic trajectory simulation, and online SGD weight updates
│   ├── recommender.py                 # Food catalog loader and personalized tolerance tiering
│   └── seed_data.py                   # Generates realistic patient history and longitudinal HbA1c trajectory
├── models/
│   ├── demo_model.joblib              # Model 1: OGTT Clearance & Remission Classifier (Real Clinical Data)
│   ├── demo_metrics.txt               # Model 1: Performance metrics, classification report & confusion matrix
│   ├── food_impact_model.joblib       # Model 2: Post-Challenge Glycemic Excursion Classifier (Real Clinical Data)
│   ├── food_impact_metrics.txt        # Model 2: Performance metrics, classification report & confusion matrix
│   └── model_metadata.json            # Machine-readable metadata, feature schemas, and provenance
├── data/
│   ├── journal.pone.0302777.s001.xlsx # Real clinical dataset (400 T2D patients, PLOS ONE 2024)
│   ├── indian_food.csv                # Comprehensive nutrient and digestion dataset (>1,100 dishes)
│   └── local/
│       └── diabetes_twin.sqlite3      # Persistent SQLite database
├── tests/
│   ├── test_diabetes_twin.py          # Unit test suite for DB, recommender, evaluator, intolerance & learning
│   ├── test_train_model.py            # Unit test suite for real-data ML loading, inference, and metadata
│   └── test_app_integration.py        # End-to-end integration and app import tests
├── requirements.txt                   # Project dependencies (Streamlit, Pandas, Scikit-learn, etc.)
├── run.bat                            # Windows Command Prompt launcher
├── run.ps1                            # Windows PowerShell launcher
└── README.md
```

---

## 🔄 End-to-End Machine Learning Model Workflow

The project implements a **dual machine learning framework**:
1. **Offline Population-Scale Clinical Models** (`train_model.py`): Trained on real patient cohort data from PLOS ONE to predict standard diagnostic remission clearance and post-challenge excursion risks.
2. **Online Adaptive Patient-Specific Twin** (`twin/online_model.py`): Real-time parameter adaptation updating an in-silico kinetic model after every meal logged by the patient.

```
       ┌─────────────────────────────────────────────────────────────┐
       │     Real Clinical Dataset (PLOS ONE journal.pone.0302777)   │
       │           400 Type 2 Diabetes Patients (Excel)              │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │         Data Hygiene & Clinical Feature Engineering         │
       │  • Weight loss (kg, %)           • BMI delta                │
       │  • HbA1c & FBG deltas            • Medication indicators    │
       │  • Negative HOMA-Beta cleaning   • Median Imputation        │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
       ┌─────────────────────────────┐ ┌─────────────────────────────┐
       │     Model 1 (demo_model)    │ │ Model 2 (food_impact_model) │
       │     OGTT Remission Cleared  │ │ Post-Challenge Excursion    │
       │     (23 Clinical Features)  │ │   (11 Clinical Features)    │
       │        ROC AUC: 0.8449      │ │       ROC AUC: 0.6833       │
       └──────────────┬──────────────┘ └──────────────┬──────────────┘
                      │                               │
                      └───────────────┬───────────────┘
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │                 Artifact Persistence in /models             │
       │  • demo_model.joblib            • food_impact_model.joblib  │
       │  • demo_metrics.txt             • food_impact_metrics.txt   │
       │  • model_metadata.json (Machine-readable schema & metrics)  │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │          Interactive Streamlit Dashboard (app.py)           │
       │  • Tab 1: Longitudinal HbA1c & Time in Range Metrics        │
       │  • Tab 2: Personalized Allowed / Restricted Food Catalog   │
       │  • Tab 3: Post-Meal Excursion & Emerging Intolerance Alert  │
       │  • Tab 4: Twin Learned Weights + Real Clinical ML Simulator │
       │  • Tab 5: Complete Clinical Journal & History Audit Log     │
       └─────────────────────────────────────────────────────────────┘
```

### Step 1: Real Clinical Dataset Ingestion
- **File**: `data/journal.pone.0302777.s001.xlsx`
- **Reference**: Tripathi, P., Kadam, N., et al. (2024). *"Oral glucose tolerance test clearance in type 2 diabetes patients who underwent remission following intense lifestyle modification: A quasi-experimental study"*, **PLOS ONE** 19(5): e0302777. [doi:10.1371/journal.pone.0302777](https://doi.org/10.1371/journal.pone.0302777)
- **Population**: 400 individuals diagnosed with Type 2 Diabetes evaluated with a 75g Oral Glucose Tolerance Test (OGTT).

### Step 2: Clinical Feature Engineering
1. **Weight & Anthropometric Dynamics**:
   $$\text{Weight Loss (kg)} = \text{Weight}_{\text{baseline}} - \text{Weight}_{\text{endline}}$$
   $$\text{Weight Loss (\%)} = \frac{\text{Weight Loss (kg)}}{\text{Weight}_{\text{baseline}}} \times 100$$
   $$\Delta\text{BMI} = \text{BMI}_{\text{baseline}} - \text{BMI}_{\text{endline}}$$
2. **Glycemic & Metabolic Shifts**:
   $$\Delta\text{HbA1c} = \text{HbA1c}_{\text{baseline}} - \text{HbA1c}_{\text{endline}}$$
   $$\Delta\text{FBG} = \text{FBG}_{\text{baseline}} - \text{FBG}_{\text{endline}}$$
3. **Data Cleaning**:
   - Standardized binary medication fields (`On_Insulin_yn`, `On_Diabetes_Medicine_yn`).
   - Cleaned anomalous negative values in HOMA equations resulting from fasting glucose dropping below $63$ mg/dL ($3.5$ mmol/L).

### Step 3: Pipeline & Model Training
Both models are trained using a reproducible scikit-learn `Pipeline`:
1. **SimpleImputer**: Median imputation for missing clinical measurements.
2. **StandardScaler**: $z$-score feature standardization ($z = \frac{x - \mu}{\sigma}$).
3. **LogisticRegression**: Balanced class weighting (`class_weight="balanced"`), L2 regularization, $2,000$ iterations, `random_state=42`.
4. **Stratified Split**: 80% training ($N=320$), 20% independent testing ($N=80$), stratified by target label.

---

## 📊 Evaluation Metrics & Performance Benchmarks

### Model 1: Oral Glucose Tolerance Test (OGTT) Clearance & Remission Classifier
- **Artifact**: `models/demo_model.joblib`
- **Metrics File**: `models/demo_metrics.txt`
- **Clinical Target**: Diagnostic remission under standard 75g OGTT criteria (`ReportInterpretation == 'GTT Cleared'`: Fasting glucose $< 100$ mg/dL and 2-hour post-load glucose $< 140$ mg/dL).
- **Cohort Distribution**: $207$ GTT Cleared ($51.8\%$) vs. $193$ Impaired / Not Cleared ($48.2\%$).

#### Performance Metrics (Test Set $N=80$):
| Metric | Impaired / Not Cleared (Class 0) | GTT Cleared / Remission (Class 1) | Overall / Macro Avg | Weighted Avg |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | `0.74` | `0.79` | `0.76` | `0.76` |
| **Recall** | `0.79` | `0.73` | `0.76` | `0.76` |
| **F1-Score** | `0.77` | `0.76` | `0.76` | `0.76` |
| **Support** | `39` | `41` | `80` | `80` |
| **Overall Accuracy** | — | — | **`76.0%`** | — |
| **ROC AUC** | — | — | **`0.8449`** | — |

#### Confusion Matrix:
```
                    Predicted: Impaired    Predicted: GTT Cleared
Actual: Impaired             31                      8
Actual: GTT Cleared          11                     30
```

#### Key Clinical Coefficients (Standardized Feature Importance):
- **Positive Predictors for Remission**:
  - $\beta$-cell recovery ($\text{HOMA-Beta}_{\text{endline}}$): **$+0.883$**
  - Baseline Insulin Resistance ($\text{HOMA-IR}_{\text{baseline}}$): **$+0.869$**
  - Degree of BMI reduction ($\Delta\text{BMI}$): **$+0.625$**
  - Baseline $\beta$-cell function ($\text{HOMA-Beta}_{\text{baseline}}$): **$+0.335$**
- **Negative Predictors for Remission**:
  - Hyperinsulinemia ($\text{Insulin}_{\text{baseline}}$): **$-1.141$**
  - High endline fasting glucose ($\text{FBG}_{\text{endline}}$): **$-0.515$**
  - Higher current HbA1c ($\text{HbA1c}_{\text{endline}}$): **$-0.404$**
  - Longer duration of diabetes ($\text{Duration}$): **$-0.276$**

---

### Model 2: Post-Challenge Glycemic Excursion & High Impact Classifier
- **Artifact**: `models/food_impact_model.joblib`
- **Metrics File**: `models/food_impact_metrics.txt`
- **Clinical Target**: Post-load glycemic spike risk following an oral carbohydrate challenge (`GTT_PP2 >= 140 mg/dL`).
- **Cohort Distribution**: $298$ Controlled ($74.5\%$) vs. $102$ High Excursion ($25.5\%$).

#### Performance Metrics (Test Set $N=80$):
| Metric | Controlled ($< 140$ mg/dL) | High Excursion ($\ge 140$ mg/dL) | Overall / Macro Avg | Weighted Avg |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | `0.82` | `0.33` | `0.57` | `0.70` |
| **Recall** | `0.55` | `0.65` | `0.60` | `0.57` |
| **F1-Score** | `0.66` | `0.43` | `0.55` | `0.60` |
| **Support** | `60` | `20` | `80` | `80` |
| **Overall Accuracy** | — | — | **`57.0%`** | — |
| **ROC AUC** | — | — | **`0.6833`** | — |

#### Confusion Matrix:
```
                    Predicted: Controlled    Predicted: High Excursion
Actual: Controlled           33                        27
Actual: High Excursion        7                        13
```

#### Key Clinical Coefficients:
- **Excursion Risk Drivers**: Pre-challenge fasting glucose ($\text{GTT\_FBSL}$: **$+0.562$**), Fasting insulin (**$+0.424$**), HbA1c (**$+0.393$**), HOMA-IR (**$+0.191$**).
- **Protective Factors**: Preserved pancreatic $\beta$-cell reserve ($\text{HOMA-Beta}_{\text{endline}}$: **$-0.937$**).

---

## 💻 Complete Run Commands & Execution Guide

### 1. Prerequisites
- **Python**: Version `3.10`, `3.11`, `3.12`, `3.13`, or `3.14`
- **Git**

### 2. Environment Setup

#### Windows (PowerShell):
```powershell
# Clone the repository (if not already local)
git clone https://github.com/RishGov1/Food_Glucose_Diabetes_ML_Project.git
cd Food_Glucose_Diabetes_ML_Project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install all required packages
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows (Command Prompt):
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

#### macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Re-train the Machine Learning Models
To train the models from scratch on the real clinical dataset (`journal.pone.0302777.s001.xlsx`):
```powershell
python train_model.py
```

**Expected Terminal Output:**
```text
=================================================================
Training Machine Learning Models on Real Clinical Dataset
Source: journal.pone.0302777.s001.xlsx
Study: PLOS ONE 19(5): e0302777 (Tripathi et al., 2024)
=================================================================
Loaded 400 patient records with 30 columns.
OGTT Remission Distribution: {1: 207, 0: 193}
Post-Challenge Excursion Distribution: {0: 298, 1: 102}
-----------------------------------------------------------------
Saved models\demo_model.joblib (ROC AUC: 0.8449)
Saved models\food_impact_model.joblib (ROC AUC: 0.6833)
Saved metadata to models\model_metadata.json
=================================================================
Model training on real clinical data complete!
```

---

### 4. Run the Streamlit Web Application

#### Option A: Direct Streamlit CLI (Recommended)
```powershell
streamlit run app.py
```
*Or using the Python module syntax:*
```powershell
python -m streamlit run app.py
```

#### Option B: Windows Shortcut Scripts
- **PowerShell**: `.\run.ps1`
- **Command Prompt**: `run.bat`

Once started, open your web browser at:
👉 **`http://localhost:8501`** (or `http://127.0.0.1:8501`)

---

### 5. Run the Automated Test Suite
To verify the database, food recommender, real-data ML models, and Streamlit integration:
```powershell
python -m pytest -v
```

**Test Coverage Summary (15 Passed Tests):**
- `tests/test_diabetes_twin.py`:
  - `test_database_initialization` — SQLite schema and default profile verification.
  - `test_seed_database_and_history` — Longitudinal history, HbA1c trajectory, and meal records.
  - `test_recommender_classification` — Allowed, Caution, and Not Recommended tiering.
  - `test_evaluator_verdicts` — Clinical classification (Good vs. Bad excursion).
  - `test_emerging_intolerance_detection` — Detection of degrading food tolerance.
  - `test_online_weight_adaptation` — SGD update step with physiological clamping.
  - `test_modeled_curve_generation` — 7-point kinetic glucose simulation curve.
- `tests/test_train_model.py`:
  - `test_clinical_dataset_exists_and_loads` — Real Excel data hygiene and feature checks.
  - `test_saved_models_exist` — Model file generation and metrics presence.
  - `test_remission_model_inference` — Live prediction test on `demo_model.joblib`.
  - `test_post_challenge_excursion_model_inference` — Live prediction test on `food_impact_model.joblib`.
  - `test_model_metadata_structure` — Schema validation of `model_metadata.json`.
- `tests/test_app_integration.py`:
  - `test_catalog_query_performance` — Food recommender speed and format.
  - `test_patient_stats_integrity` — Longitudinal summary calculations.
  - `test_app_import_and_schema` — Headless app execution and module import check.

---

### 6. Reset / Reseed the Database (Optional)
To reset the SQLite database to the initial 30 days of longitudinal history:
```powershell
python -c "from twin.seed_data import seed_database; seed_database(force_reseed=True)"
```
*(You can also click the **"🔄 Reset Demo Database"** button in the app's sidebar).*

---

## 📋 Interactive Verification Walkthroughs

### Walkthrough 1: Real Clinical Patient Remission Simulator
1. Open the application and switch to the **"🧠 Adaptive Model & Weights"** tab.
2. Scroll to the **"🔬 Clinical Machine Learning Models (PLOS ONE Real Patient Cohort)"** section.
3. Open the **"🩺 Run Patient Clinical Prediction Simulator"** expander.
4. Enter patient parameters:
   - **Diabetes Duration**: `4.5` years
   - **Baseline Weight**: `78.0` kg $\rightarrow$ **Current Weight**: `72.0` kg ($-7.7\%$ weight reduction)
   - **Baseline HbA1c**: `8.2%` $\rightarrow$ **Current HbA1c**: `6.8%`
   - **Current Fasting Glucose**: `104` mg/dL
5. Click **"🔮 Evaluate Patient Remission & Post-Challenge Risk"**.
6. **Result**:
   - The real ML model calculates an **OGTT Remission Clearance Probability of ~78%**.
   - Identifies the $6.0$ kg weight loss and HbA1c normalization as the primary statistical drivers.

---

### Walkthrough 2: Emerging Food Intolerance Detection
1. Switch to the **"🍽️ Log Meal & Real-Time Evaluation"** tab.
2. Select **`Sweetened Instant Oats with Milk`**.
3. In the patient's historical profile, this food was previously well-tolerated with small spikes ($+24$ to $+26$ mg/dL).
4. Enter pre-meal glucose `120 mg/dL` and post-meal readings peaking at `178 mg/dL` ($+58$ mg/dL excursion).
5. Click **"Log Meal & Run Digital Twin Analysis"**.
6. **Result**:
   - Verdict: 🔴 **BAD — Glycemic Spike / Excursion**
   - Alert: ⚠️ **Emerging Intolerance Alert!** (*"Developing Intolerance Detected! Historically, your Digital Twin recorded a safe average rise of +25 mg/dL. Today's rise of +58 mg/dL shows developing metabolic sensitivity."*)
   - Telemetry: **Carb Sensitivity weight adaptively increments** from `0.88` to `0.92` mg/dL/g, and the food is downgraded to *Not Recommended* in the food intelligence catalog.

---

## ⚖️ Educational & Research Disclaimer
This application is an educational prototype and in-silico simulation demonstrating personalized digital twin architectures, online recursive learning, and predictive clinical modeling based on published medical literature. It does not provide medical diagnoses, insulin titration algorithms, or clinical prescriptions. Consult certified endocrinologists and healthcare providers for clinical medical management.

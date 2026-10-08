# Food + Postprandial Glucose Diabetes Risk ML Project

## Project goal

Build a machine-learning system that analyzes food intake and post-meal glucose behavior and estimates whether a person belongs to a diabetes-positive or diabetes-negative class.

The intended user workflow is:

1. Enter the glucose measurements immediately after a meal, 15 min, 40 min, 45 min, 1 h and 2 h after the meal.
2. Enter optional meal/macronutrient information.
3. Compute postprandial response features such as peak glucose, glucose excursion, recovery, and AUC.
4. Produce a diabetes-risk probability and response summary.
5. Evaluate the cross-dataset diagnostic signal against the PIMA Indians Diabetes dataset.

## Important scientific limitation

The requested datasets do **not** contain the same six glucose measurements.

D1NAMO contains 20 healthy participants and 9 participants with Type 1 diabetes. The diabetic subset includes continuous glucose monitoring (CGM), while healthy participants have much sparser/manual glucose measurements around meals. The official D1NAMO description also states that its CGM processing uses 5-minute glucose sections.

PIMA contains 768 records and eight diagnostic variables plus the outcome. Its `Glucose` feature is plasma glucose concentration from a 2-hour oral glucose tolerance test. It does not contain 0/15/40/45/60/120-minute post-meal glucose curves.

Therefore, this project intentionally uses **two related models**:

### Model A — D1NAMO meal-response model

This is the main subject-aware classifier. It uses features that can be constructed across the D1NAMO healthy and diabetic cohorts, such as:

- glucose near meal start
- 2-hour glucose
- 2-hour glucose change
- available meal-intake metadata/macronutrients

The six entered readings are still fully analyzed in the application and are converted into response descriptors.

### Model B — D1NAMO → PIMA external validation model

For a fair external test, a separate glucose-only classifier is trained on D1NAMO's 2-hour glucose measurement and then evaluated on PIMA's `Glucose` column.

This is **not** claimed to be validation of a six-point post-meal classifier. PIMA's 2-hour OGTT glucose is a different measurement protocol from a 2-hour post-meal CGM value. It is a cross-dataset stress test of how much the glucose level alone transfers.

The code will not invent the five missing PIMA timepoints.

## Dataset sources

### D1NAMO

Official Zenodo record:
https://zenodo.org/records/5651217

Project repository:
https://github.com/PSI-TAMU/D1NAMO

### PIMA Indians Diabetes

UCI:
https://archive.ics.uci.edu/dataset/34/pima+indians+diabetes

A commonly used CSV mirror has columns:
`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

## Data layout

Extract the D1NAMO archive so the project can find the participant folders.

Recommended layout:

```text
data/raw/d1namo/
├── healthy_subset_pictures-glucose-food/
│   ├── 001/
│   │   └── ...
│   └── ...
└── diabetes_subset_pictures-glucose-food-insulin/
    ├── 001/
    │   ├── glucose.csv
    │   ├── insulin.csv
    │   └── ...
    └── ...
```

The loader searches recursively, so exact parent naming can vary as long as `healthy` or `diabetes` is present somewhere in the path.

Put PIMA here:

```text
data/raw/pima/diabetes.csv
```

Do not commit the large D1NAMO archives to GitHub.

## Installation

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the complete pipeline

```bash
python run_all.py
```

This will:

1. discover D1NAMO participant files
2. detect meal events
3. align glucose around each meal
4. save `outputs/d1namo_meal_samples.csv`
5. perform subject-aware cross-validation
6. train the D1NAMO common-feature classifier
7. train the D1NAMO glucose-only model
8. evaluate the glucose-only model on PIMA
9. save metrics and figures

## Run the dashboard

```bash
streamlit run app.py
```

Then open the local Streamlit URL.

## What the dashboard calculates from the six readings

For:

- 0 min
- 15 min
- 40 min
- 45 min
- 60 min
- 120 min

it computes:

- baseline/initial glucose
- peak glucose
- time of peak
- peak excursion
- 2-hour excursion
- trapezoidal AUC
- 60-to-120 minute recovery
- 0-to-15 slope
- 60-to-120 slope
- percentage peak rise

These are useful for analyzing postprandial response even when the cross-dataset diagnostic classifier cannot consume all six values.

## Evaluation

The project reports:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- ROC-AUC where valid
- confusion matrix
- ROC curve
- precision-recall curve
- subject-aware validation results

PIMA is used as an **external evaluation dataset**, not as a training dataset.

## Leakage prevention

Meal observations from the same person are highly correlated. A random row-level train/test split would leak subject-specific patterns.

This project uses `StratifiedGroupKFold` with participant ID as the group, so a participant's meals are kept inside the same fold.

## Medical disclaimer

This is an academic ML project, not a medical diagnostic device. The model output is a statistical estimate and must not be used as a diagnosis or as a substitute for clinical testing.
"# Food_Glucose_Diabetes_ML_Project" 

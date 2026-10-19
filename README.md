# Food-Aware Post-Meal Glucose ML Demo

## Important medical limitation
This is an **educational prototype only**, not a Type 1 diabetes diagnostic model. It was trained on synthetic generated examples with invented labels, not clinical patient data. The displayed class and score have no validated clinical meaning and must not be used to make medical decisions.

Type 1 diabetes diagnosis requires appropriate clinical assessment and validated tests interpreted by a qualified healthcare professional. Six post-meal readings plus meal details cannot reliably diagnose or rule out Type 1 diabetes.

## What changed in this version
The model now takes **both glucose readings and selected food features**:
- Glucose at 0, 15, 30, 45, 60, and 120 minutes
- Estimated carbohydrates (g) for the selected portion
- Estimated fibre (g)
- Estimated fat (g)
- Estimated protein (g)
- Estimated digestion time (hours)

The app reads nutritional values from `data/indian_food.csv` and scales them by the selected number of servings. The food's name is used to look up its nutritional features; the raw name itself is not a separate model feature. Two foods with similar nutrition values may therefore produce similar features.

## Included files
- `app.py`: Streamlit web app with food selection, servings, nutrition estimates, glucose inputs and model output.
- `train_model.py`: reproducible training script.
- `data/indian_food.csv`: copy of the supplied Indian food dataset (1,137 records).
- `data/digestion_notes.csv`: notes sheet from the supplied workbook.
- `data/synthetic_glucose_demo.csv`: 6,000 synthetic demonstration records, balanced between two invented classes.
- `models/demo_model.joblib`: pre-trained food-aware model.
- `models/demo_metrics.txt`: metrics on synthetic data only.

## Setup on Windows
1. Install Python 3.10 or 3.11 from https://www.python.org/downloads/ . Enable **Add Python to PATH** during installation.
2. Extract the ZIP.
3. Open PowerShell or Command Prompt in the extracted project folder.
4. Create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   ```
5. Activate it in PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   Or in Command Prompt:
   ```bat
   .venv\Scripts\activate.bat
   ```
6. Install packages:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Launch:
   ```bash
   streamlit run app.py
   ```
8. If the browser does not open, visit `http://localhost:8501`.

## Retrain the model
With the virtual environment activated:
```bash
python train_model.py
```
This retrains on the included synthetic labels. It does not make the model clinically valid.

## Bias and data quality
- The synthetic dataset has 3,000 examples per invented class.
- Demographic variables such as age, sex, ethnicity, and location are not included.
- Balanced class counts do not prove that data are unbiased or representative.
- Synthetic labels/trajectories are invented assumptions and may embed unrealistic relationships.
- Metrics from this data measure only how well the model reproduces the synthetic labeling process; do not report them as clinical accuracy.

## Food feature limitations
The supplied food workbook has estimated nutrient and digestion fields. The app scales nutrients linearly by servings and converts digestion-time ranges to their midpoint. These are approximate dataset features, not a model of an individual's digestion or glucose response. Mixed meals, preparation, medications, insulin, activity, stress, illness and measurement devices can affect readings.

## What a valid research version would require
1. Ethically approved, consented, de-identified glucose time-series and meal records paired with clinically verified diagnoses.
2. A clearly defined target and inclusion criteria, with relevant meal, portion, medication, device, and participant metadata.
3. Patient-level train/validation/test separation to prevent leakage between repeated readings from one person.
4. External validation, calibration, subgroup fairness evaluation, and clinician oversight before any clinical use.

# Glucose Twin

A local Streamlit app for exploring Indian meals, recording glucose responses and building a personal food history. It has a trained meal-response model, a separate observed-glucose benchmark, and a measured-outcome learning loop.

**This is a research prototype.** The initial meal model is trained on explicitly simulated responses. It cannot establish that a food is medically safe, diagnose diabetes, or guide insulin or medication. The real glucose benchmark is small and does not demonstrate reliable two-hour prediction. All results and limitations are available inside **Model & data** and in [MODEL_REPORT.md](MODEL_REPORT.md).

## Open the app

From PowerShell in this project:

```powershell
.\start_app.ps1
```

Open **http://127.0.0.1:8501**. The app binds to this computer only. Stop it with Ctrl+C when launched in your terminal.

The current workspace already includes the prepared dataset and trained models. It does not need to retrain when the app opens.

## What you can do

1. **Meal check:** enter a premeal glucose reading in mg/dL or mmol/L, select foods and weighed portions, and compare illustrative 60- and 120-minute estimates. Nutrition and source information stay visible.
2. **My food lists:** search foods and filter personal assessments. “Likely suitable” is a provisional allowed list; “Review portion” and “Limit / discuss” are review lists. A cold-start estimate remains unconfirmed. A food is never permanently banned.
3. **Log a meal:** select one or several foods you ate, enter their weights, your premeal reading and the meal start date/time, and save. You are taken directly to the journal; checking a prediction first is not required.
4. **Meal journal:** select the saved meal and add glucose at **15, 30, 45, 60 minutes (1 hour), and 120 minutes (2 hours)**. Save one reading at a time or several together. Leave unavailable readings blank and return later. Saved readings are protected from overwriting. A meal-specific chart and the history export include all entered timepoints and actual elapsed times.
5. **Add a food:** enter a label or recipe calculation per 100 g. Custom foods are private to the selected local profile. New food outcomes participate in the same measured-only learning loop.
6. **Model & data:** inspect the model comparison, separate observed benchmark, provenance, bias limitations and full report.

Start with the example profile to explore. Create a personal profile before recording your own meals. Default readings are examples until you replace them. Use your clinician's targets in the sidebar. Allergy and care notes are reminders; the app does not automatically check them.

The 15/30/45-minute readings are stored and plotted; the current personal model learns the 60/120-minute targets after a two-hour outcome is available. For learning, the two-hour measurement must be taken 105–135 minutes after the meal began, and a supplied one-hour measurement must be at 50–70 minutes. Outside-window values stay in history but exclude the meal from training. The app needs at least 12 eligible outcomes and four later validation meals before it activates a personal correction. An update must reduce later-meal error by at least 5% and 1 mg/dL. Each target's recording time is tracked separately so backfilling an earlier reading cannot create prospective validation evidence.

## Data expansion

The original `data/` files are preserved. The full Indian Nutrient Databank was downloaded separately and merged with the original workbook, keeping source labels and a duplicate audit. Generated outputs include:

- `data/processed/foods.csv`: 2,140 distinct normalized foods, including 1,189 recipes and fruit entries eligible for meal scenarios. Other ingredients remain available for composition reference.
- `data/processed/food_portions.csv`: 10,700 calculated portion rows, not 10,700 different foods.
- `data/processed/synthetic_meals.csv.gz`: the 100,000 simulated meal scenarios you selected. Every row explicitly says it is synthetic.
- `data/processed/quarantined_foods.csv`: seven source rows with invalid nutrient values, retained for review.
- `reports/data_audit.json`: counts, provenance and source hash.

Source: [Indian Nutrient Databank](https://www.anuvaad.org.in/indian-nutrient-databank/) and its [methodology](https://www.anuvaad.org.in/methodology/). The source is nutrition information, not measured Indian patient glucose responses. The expansion does not establish that population bias has been removed.

## Rebuild from the preserved data

For a new environment, Python 3.11 or later is recommended. This build was verified on Python 3.13 on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Recreate the prepared data, compare and fit models, evaluate the observed data, and refresh the report:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --rows 100000
.\.venv\Scripts\python.exe scripts\train_models.py
.\.venv\Scripts\python.exe scripts\evaluate_observed.py
.\.venv\Scripts\python.exe scripts\write_report.py
```

The preparation script downloads the public INDB workbook only if the cached source is absent. Existing D1NAMO data must be present under `data/raw/d1namo/`; it is not redownloaded. Normal app use works offline. CPU training is sufficient for these models; no GPU or NPU is required.

Do not rerun preparation while using the app. Rebuilding the base model changes its version; older personal outcomes remain in the journal but do not train a correction against a different base version. Keep a copy of previous model artifacts if maintaining a longitudinal study. Use only trusted local joblib model files.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=reports\test_tmp
```

Tests cover data preservation, source/split integrity, units, out-of-range abstention, profile isolation, timing and duplicate protection, measured-only learning, chronological validation, stale results and Streamlit workflows. Browser screenshots and checks are in `reports/`. Optional browser verification uses `requirements-dev.txt` and a locally installed Chrome browser:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts\browser_check.py
```

## Project layout and storage

```text
app.py                         Streamlit entry point
twin/ui.py                     Six app pages and local-profile controls
twin/service.py                Nutrition, prediction, evidence and safety rules
twin/learning.py               Validation-gated personal residual models
twin/storage.py                SQLite persistence and input constraints
twin/estimators.py             Reproducible base-estimator definitions
scripts/prepare_data.py        Source audit, food merge, portion and response simulation
scripts/train_models.py        Seven-candidate model comparison and calibration
scripts/evaluate_observed.py   Separate participant-held-out D1NAMO benchmark
scripts/write_report.py        Report generated from actual experiment results
models/                        Trained models used by the app
data/local/twin_v2.sqlite3      New local profiles, foods, meals and learning audit
data/local/meals.sqlite3        Preserved legacy database; not imported or modified
```

Profiles separate records within the app but are **not authentication**. Anyone with access to the computer can select a profile. The database is unencrypted. Keep the app on localhost and use a trusted personal device. It does not upload patient data, sync to a cloud or use an external AI API. Streamlit usage statistics are disabled. Exported CSV cells are escaped against spreadsheet formula injection.

The previous implementation was moved to `D:\Food_Glucose_Diabetes_ML_Project_legacy_20261018` before rebuilding. The original workbook, raw datasets and legacy meal database remain in this project's `data/` directory. SHA-256 checks for all 446 pre-existing data files are recorded in `reports/original_data_manifest.json` and verified by tests.

Dataset attribution and upstream usage terms remain applicable. This project does not claim ownership of the original INDB, IFCT or D1NAMO data.

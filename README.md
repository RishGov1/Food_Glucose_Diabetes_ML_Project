# GlucoMeal: a personal glucose twin prototype

Local Streamlit app trained on the supplied D1NAMO meal/glucose data. It forecasts 60- and 120-minute glucose from readings at 0, 15, **40**, and 45 minutes; reviews foods and portions; and updates a separate personal correction when actual outcomes are saved.

**Research use only.** This is not a clinically validated digital twin, diabetes diagnostic tool, or insulin/dosing system. The selected model did not outperform the simple last-reading baseline on unseen participants. Its two-hour predictions are particularly uncertain.

## Start

From this directory in PowerShell:

```powershell
.\start_app.ps1
```

Open http://127.0.0.1:8501. The server binds only to localhost and disables Streamlit usage statistics. Dependencies and trained artifacts are already installed/generated on this laptop.

For a fresh environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -u run_all.py
```

GPU training uses PyOpenCL with the installed Intel graphics driver. NPU inference uses OpenVINO and the Intel AI Boost driver. No PyTorch/CUDA installation is required. Training fails explicitly if a GPU is unavailable instead of claiming CPU work was GPU training.

## Models trained on 12 October 2026

| Model | Purpose | Device |
|---|---|---|
| Extra Trees residual regressor, 250 trees | Selected population forecast; predicts change from the 45-minute reading | CPU |
| Ridge residual regressor | Forecast candidate | CPU |
| Random forest residual regressor, 250 trees | Forecast candidate | CPU |
| Dense neural network, 4 → 32 → 16 → 2, ReLU, Adam, 350 epochs | Forecast candidate; exported to OpenVINO | Intel Arc 130T GPU training |
| Persistence | Carry forward the last reading; comparison baseline | CPU |
| Shrunk personal offset, then validation-gated ridge | Automatically learns from each profile's measured meals | CPU |
| Logistic regression cohort classifier | Historical six-point healthy-vs-Type-1 research classification | CPU |
| Random forest glucose-only classifier | D1NAMO-to-PIMA protocol stress test | CPU |

The NPU performed **inference**, not training. CPU, GPU and NPU inference all passed numerical comparison against the saved neural network on six curves. OpenVINO exports are `models/meal_neural.xml` and `.bin`; fitted models are `.joblib` files. `outputs/hardware.json` records actual device names, timings and numerical differences.

## Validation results

There are 387 meal events from 27 usable participants, but only **60 complete measured curves from 7 Type 1 participants** for forecasting. Future outcomes are never imputed as training labels. Nearest measurements use the configured eight-minute tolerance; all early inputs are restricted to measurements available by minute 45. This timing approximation and occasionally reused nearby readings remain limitations.

Five participants (37 meals) form the training/selection set. Grouped CV keeps each participant in one fold. Two other participants (23 meals) form an untouched holdout. Model selection uses only training CV; the holdout is evaluated once. Deployment models are then refitted on all 60 curves.

| Candidate | Training grouped-CV MAE, mg/dL |
|---|---:|
| Extra Trees | 23.55 |
| Ridge | 23.67 |
| Random forest | 23.83 |
| Neural network | 25.12 |
| Persistence baseline | 29.51 |

Selected-model holdout MAE: **14.16 mg/dL at 60 minutes**, **52.65 at 120 minutes**, **33.41 overall**. Persistence holdout MAE: **31.19 overall**. There is no demonstrated improvement over baseline on this small holdout. The UI displays this limitation. OOF error references are descriptive error percentiles, not calibrated personal confidence intervals.

The legacy cohort classifier can exploit different measurement protocols and missingness across cohorts; its high internal classification scores are **not evidence of diagnostic accuracy**. PIMA is not meal-forecast validation: it contains OGTT glucose and a different population. Its glucose-only cross-dataset ROC-AUC was 0.7271 on 763 nonzero-glucose rows.

Data fixes include cohort-qualified participant IDs, parsing colon-separated food dates, aligning meals to their first event, excluding the ambiguous `012_diabetes` participant from the healthy class, ignoring entirely missing training rows, and treating zero PIMA glucose as missing. Source files were preserved.

## Try the frontend

Keep **Demo mode on**, choose a sample and press **Load test example**. This selects `Boiled rice (Uble chawal)` at one listed serving. Press **Forecast next readings**, then record the sample outcomes and tick the confirmation before **Save outcome & update my twin**.

All numbers below are synthetic test values in mg/dL, not medical targets.

| Example | 0 min | 15 min | 40 min | 45 min | Measured 60 min | Measured 120 min |
|---|---:|---:|---:|---:|---:|---:|
| Gentle rise | 105 | 115 | 130 | 128 | 125 | 110 |
| Larger rise | 130 | 160 | 205 | 215 | 225 | 195 |
| Slower recovery | 115 | 140 | 175 | 185 | 205 | 190 |

Before personal learning, expected forecasts for these examples are approximately **136.7/114.1**, **219.9/185.8**, and **187.9/155.1** (60/120 minutes). Subsequent predictions change as outcomes are saved. Use a different meal date/time for each distinct test meal; resaving the same time is intentionally deduplicated.

Expand **Compare the GPU-trained neural network**, select **NPU**, and press **Run neural comparison** to exercise the NPU from the frontend. That is a separately evaluated comparison model; it does not replace the main Extra Trees prediction.

## How personal learning works

Only explicitly saved measured outcomes become labels. Predictions are frozen before outcomes are entered; changing the inputs invalidates that forecast. Observations are stored in the append-only SQLite database `data/local/meals.sqlite3`, with model version, profile, foods, portions, measured values and original predictions.

The first measured meal updates a regularized personal offset. From eight meals, a ridge model uses early glucose and carbohydrate, sugar, fibre, protein and fat estimates. It is trained on earlier meals and checked on the latest quarter (at least two meals); it replaces the simple offset only after improving validation MAE by at least 5%. Personal corrections are capped at ±40 mg/dL. These controls do not establish clinical safety or prove improvement for an individual.

The personal model is reconstructed deterministically from saved observations, so it survives restarts and cannot become stale after a save. Demo and real histories are separate. Profiles are local labels, not authenticated accounts. Retraining the population model creates a new version; old history remains available but does not calibrate the new model using predictions from an old version. Back up `data/local/` and `models/` together; the database contains personal data and is not encrypted by the app.

## Foods to review

The spreadsheet has 1,137 entries, not paired food-specific glucose outcomes. The initial population model therefore does not claim to have learned effects of these Indian foods. Food-aware personal modeling starts only with the user's paired observations.

Nutrition flags per listed serving use transparent project heuristics: free sugars ≥10 g, carbohydrate ≥45 g, or carbohydrate ≥25 g with fibre <3 g. Serving units are shown because some spreadsheet entries represent large portions. These are not clinical cutoffs or universal bad-food labels. A separate personal list shows foods present in meals with measured outcomes above the chosen reference, without claiming ingredient causation.

## Tests and artifacts

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Tests cover GPU/CPU numerical agreement for backpropagation, timestamp parsing, leakage prevention, invalid inputs, duplicate protection, profile/version isolation, persistent personal updates, stale form inputs, food search, and the Streamlit NPU button. On Windows with mixed sandbox/user permissions, use a fresh `--basetemp` directory.

- `outputs/twin_metrics.json`: CV/holdout metrics and split participant IDs.
- `outputs/hardware.json`: actual hardware inference verification.
- `outputs/response_model_metrics.json`, `outputs/glucose_model_metrics.json`, `outputs/pima_external_metrics.json`: cohort/stress-test results.
- `reports/TRAINING_RUN.md`: concise run report.

## References

- [D1NAMO dataset](https://zenodo.org/records/5651217)
- [ADA: carbohydrates and glucose](https://diabetes.org/food-nutrition/understanding-carbs/get-to-know-carbs)
- [ADA: glucose monitoring and individualized targets](https://diabetes.org/living-with-diabetes/treatment-care/checking-your-blood-sugar)
- [OpenVINO inference devices](https://docs.openvino.ai/2024/about-openvino/compatibility-and-support/supported-devices.html)
- [PyOpenCL runtime](https://documen.tician.de/pyopencl/runtime_program.html)

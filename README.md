# Food-Aware Glucose Digital Twin Demo

## Important limitation
Educational prototype only. Both models are trained on synthetic examples and invented labels, not clinical patient data. The scores are not validated predictions of food effects and must not be used for medical decisions. HbA1c reflects longer-term average glucose and does not directly predict the response to one meal. Type 1 diabetes cannot be diagnosed or ruled out using this application.

## Detailed combined output (updated)
The interface now combines both model outputs into one plain-language report. It includes the food-impact category and score, a descriptive glucose-pattern label instead of showing only class 0/1, a combined agreement/mixed-result explanation, a chart and summary of the entered readings, meal nutrient context (carbohydrates, fibre, fat and protein), digestion-time caveats, and a separate explanation of HbA1c. The chart shows user-entered readings only; the app does not claim to forecast future glucose values.

## Inputs
The app now asks for:
- Current blood glucose (mg/dL)
- Most recent HbA1c (%)
- Selected food and serving count
- Nutrition estimates for the portion: carbohydrates, fibre, fat and protein
- Estimated digestion time from the food dataset
- Six glucose readings for the separate glucose-pattern model

The food-impact model returns a synthetic category: lower modeled impact, higher modeled impact, or uncertain. This category is based on an invented synthetic labeling formula, not measured human responses. It must not be interpreted as a clinical judgement that a food is safe or unsafe.

## Files
- `app.py`: Streamlit interface with detailed combined interpretation of both model outputs, glucose-reading chart, meal-nutrient context, and HbA1c explanation.
- `train_model.py`: trains both demonstration models.
- `data/indian_food.csv`: supplied Indian food dataset.
- `data/digestion_notes.csv`: workbook notes.
- `data/synthetic_glucose_demo.csv`: synthetic training data.
- `models/demo_model.joblib`: glucose-pattern demo classifier.
- `models/food_impact_model.joblib`: synthetic food-impact category classifier.
- `models/*metrics.txt`: synthetic-only metrics.

## Run on Windows
1. Install Python 3.10 or 3.11 and enable Add Python to PATH.
2. Extract the ZIP and open a terminal in the project folder.
3. Create and activate a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   In Command Prompt use `.venv\Scripts\activate.bat`.
4. Install packages:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. Open `http://localhost:8501` if it does not open automatically.
7. To retrain both synthetic demo models:
   ```bash
   python train_model.py
   ```

## Bias and validity
Synthetic labels are invented. Balanced class counts and omitting demographic variables do not prove fairness or clinical validity. Do not report synthetic metrics as medical accuracy. A research-grade version requires consented, representative real data, clinically defined outcomes, participant-level data splits, external validation, uncertainty calibration, subgroup fairness evaluation and clinical oversight.

# ML-Based Food Intake and Postprandial Glucose Analysis for Diabetes Risk Estimation

## 1. Abstract

This project develops a machine-learning framework for analyzing food intake and post-meal glucose behavior to estimate diabetes-related risk. The D1NAMO dataset is used as the primary training source because it contains glucose measurements, food annotations and subjects with both healthy and Type 1 diabetes labels. The PIMA Indians Diabetes dataset is used as an external evaluation source.

The main challenge is that the two datasets do not expose identical glucose measurements. D1NAMO contains continuous glucose monitoring for diabetic participants and sparse/manual glucose observations for healthy participants, while PIMA contains a single 2-hour oral glucose tolerance test glucose measurement. Consequently, the project separates the full six-point postprandial response analysis from the cross-dataset classifier. The deployed cross-dataset classifier uses the 2-hour glucose feature because that feature is directly available in both datasets, while the six-point interface calculates additional response descriptors such as peak glucose, excursion, slopes, recovery and area under the curve.

## 2. Objectives

1. Process food and glucose observations from D1NAMO.
2. Align glucose observations around meal timestamps.
3. Create postprandial glucose response features.
4. Prevent participant-level leakage using group-aware validation.
5. Train a D1NAMO diabetes-risk classifier.
6. Train a PIMA-compatible glucose-only model on D1NAMO.
7. Evaluate the glucose-only model externally on PIMA.
8. Provide a user-facing dashboard.

## 3. System Architecture

```text
                 D1NAMO
       ┌──────────┴──────────┐
       │                     │
   Food/meal data        Glucose data
       │                     │
       └──────────┬──────────┘
                  │
          Meal-time alignment
                  │
       ┌──────────┴──────────┐
       │                     │
 Six-point response      Common features
  analysis layer        g0, g120, delta120
       │                     │
       │              D1NAMO classifier
       │                     │
       │               saved model
       │
       └──────────────┐
                      │
                Streamlit app
                      │
       ┌──────────────┴─────────────┐
       │                            │
Six-point response             Risk estimate
visualization                  from compatible
                               glucose model
                                      │
                                   PIMA test
```

## 4. D1NAMO

D1NAMO is an open multimodal diabetes-management dataset. It contains 20 healthy participants and 9 participants with Type 1 diabetes and includes glucose measurements, wearable signals, and food-related annotations.

The original archive is large, so it should be downloaded separately and placed in `data/raw/d1namo/`.

## 5. PIMA

The PIMA dataset contains 768 observations. Its input fields include:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

The target is `Outcome`.

The `Glucose` field is the plasma glucose concentration obtained from a 2-hour oral glucose tolerance test.

## 6. Feature engineering

For the six-point input:

t = [0, 15, 40, 45, 60, 120]

let the glucose values be:

G0, G15, G40, G45, G60, G120.

### Peak glucose

Gpeak = max(G0, G15, G40, G45, G60, G120)

### Peak excursion

Delta_peak = Gpeak - G0

### Two-hour excursion

Delta_120 = G120 - G0

### Response area

AUC = integral from 0 to 120 of G(t) dt

In the implementation, trapezoidal numerical integration is used.

### Early slope

Slope_0_15 = (G15 - G0) / 15

### Late slope

Slope_60_120 = (G120 - G60) / 60

### Recovery

Recovery_60_120 = G60 - G120

### Percentage peak rise

PeakRise% = ((Gpeak - G0) / G0) * 100

## 7. Machine-learning approach

The main classification family contains:

- Logistic Regression with class balancing
- Random Forest with class balancing

Missing feature values are median-imputed. Logistic Regression uses standardized features.

Subject-level grouping is used in cross-validation so that multiple meals from the same participant cannot appear in both training and validation folds.

The best model is selected using ROC-AUC followed by balanced accuracy.

## 8. Cross-dataset evaluation protocol

A direct six-point D1NAMO-to-PIMA evaluation would be invalid because PIMA does not contain the six post-meal time points.

Therefore:

1. Train a glucose-only classifier on D1NAMO using its 2-hour glucose feature.
2. Map that feature to PIMA's `Glucose` column.
3. Evaluate on PIMA `Outcome`.

The result should be described as external cross-dataset stress testing, not proof that the six-point post-meal model generalizes clinically.

## 9. Evaluation metrics

The project reports:

Accuracy = correct predictions / total predictions

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = harmonic mean of precision and recall

Balanced Accuracy = average of sensitivity and specificity

ROC-AUC = area under the receiver operating characteristic curve

## 10. Expected deliverables

After execution:

```text
outputs/
├── d1namo_meal_samples.csv
├── common_model_metrics.json
├── glucose_model_metrics.json
├── pima_external_metrics.json
└── figures/
    ├── d1namo_response.png
    └── d1namo_class_distribution.png

models/
├── d1namo_common_model.joblib
├── d1namo_glucose_only_model.joblib
└── glucose_threshold.json
```

## 11. Limitations

1. D1NAMO has a very small number of participants.
2. The dataset mixes Type 1 diabetes with healthy controls, whereas PIMA is a separate population and protocol.
3. PIMA's glucose measurement is an OGTT measurement, not a post-meal CGM curve.
4. Participant-level labels are not equivalent to disease diagnosis from one meal.
5. Meal annotations and glucose timestamps can have missing or inconsistent observations.
6. The model is not validated for clinical use.

## 12. Future improvements

A stronger research version should use a dataset containing healthy, prediabetic and Type 2 participants with continuous glucose monitoring and standardized meal challenges. A six-point classifier could then be trained directly on the intended input representation without imputation or cross-protocol assumptions.

Additional modalities such as age, BMI, insulin dose, physical activity and heart-rate variability can be incorporated when a common measurement protocol is available.


## Updated model input and food database integration

The main D1NAMO classifier uses the six glucose observations (0, 15, 40, 45, 60 and 120 minutes) and derived response features such as peak, AUC, excursions, slopes and recovery. The Indian food spreadsheet is used to calculate serving-adjusted meal nutrition and to contextualize the measured response. The food spreadsheet has no participant diagnosis labels or paired glucose curves, and the D1NAMO label rows do not identify those exact foods. Therefore, the food nutrients are not represented as learned diagnostic predictors; doing so would create unsupported associations. A future genuinely food-aware supervised model requires meal-level records linking actual food identity/portion, time-aligned glucose outcomes and diagnostic labels for the same participants.

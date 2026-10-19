"""Train the food-aware educational classifier on synthetic data only."""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic_glucose_demo.csv"
MODEL = ROOT / "models" / "demo_model.joblib"
METRICS = ROOT / "models" / "demo_metrics.txt"
FEATURES = [
    "glucose_0min", "glucose_15min", "glucose_30min", "glucose_45min", "glucose_60min", "glucose_120min",
    "meal_carbs_g", "meal_fibre_g", "meal_fat_g", "meal_protein_g", "digestion_hours"
]

df = pd.read_csv(DATA)
X, y = df[FEATURES], df["synthetic_demo_label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
joblib.dump(model, MODEL)
with open(METRICS, "w", encoding="utf-8") as f:
    f.write("METRICS ON HELD-OUT SYNTHETIC DATA ONLY\n")
    f.write("Not evidence of real-world diabetes detection performance.\n\n")
    f.write(classification_report(y_test, pred, target_names=["Synthetic class 0", "Synthetic class 1"]))
    f.write(f"\nROC AUC (synthetic test split): {roc_auc_score(y_test, prob):.4f}\n")
    f.write("\nConfusion matrix:\n" + str(confusion_matrix(y_test, pred)) + "\n")
print("Saved model:", MODEL)
print("Saved metrics:", METRICS)
print("Features used:", ", ".join(FEATURES))
print("Reminder: synthetic metrics are not clinical validation.")

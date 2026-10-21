"""Train the synthetic educational models. Not for clinical use."""
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
df = pd.read_csv(ROOT / "data" / "synthetic_glucose_demo.csv")
def train_model(X, y, path, metrics_path, names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    joblib.dump(model, path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("SYNTHETIC DATA METRICS ONLY. Not evidence of clinical performance.\\n\\n")
        f.write(classification_report(y_test, pred, target_names=names))
        f.write(f"\\nROC AUC: {roc_auc_score(y_test, prob):.4f}\\n\\nConfusion matrix:\\n{confusion_matrix(y_test, pred)}\\n")
    print("Saved", path)
pattern_features = ["glucose_0min","glucose_15min","glucose_30min","glucose_45min","glucose_60min","glucose_120min",
                    "current_glucose_mgdl","hba1c_percent","meal_carbs_g","meal_fibre_g","meal_fat_g","meal_protein_g","digestion_hours"]
impact_features = ["current_glucose_mgdl","hba1c_percent","meal_carbs_g","meal_fibre_g","meal_fat_g","meal_protein_g","digestion_hours"]
train_model(df[pattern_features], df["synthetic_demo_label"], ROOT/"models"/"demo_model.joblib",
            ROOT/"models"/"demo_metrics.txt", ["Synthetic class 0","Synthetic class 1"])
train_model(df[impact_features], df["synthetic_food_impact_label"], ROOT/"models"/"food_impact_model.joblib",
            ROOT/"models"/"food_impact_metrics.txt", ["Lower synthetic modeled impact","Higher synthetic modeled impact"])

"""Configuration constants and clinical thresholds for Diabetes Digital Twin."""
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
LOCAL_DATA_DIR = DATA_DIR / "local"
FOOD_DATASET_PATH = DATA_DIR / "indian_food.csv"
DB_PATH = LOCAL_DATA_DIR / "diabetes_twin.sqlite3"

# Ensure local storage directory exists
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Clinical Thresholds (Standard ADA / RSSDI Guidelines)
TARGET_MIN_GLUCOSE = 70.0    # mg/dL (Hypoglycemia floor)
TARGET_MAX_GLUCOSE = 160.0   # mg/dL (Post-meal upper target for T2D)
TARGET_FASTING_MAX = 130.0   # mg/dL

# Post-Meal Glycemic Verdict Thresholds
SPIKE_GOOD_MAX = 35.0        # Delta rise <= 35 mg/dL is considered optimal/good
SPIKE_MODERATE_MAX = 55.0    # Delta rise between 35 and 55 mg/dL is moderate/borderline
# Delta rise > 55 mg/dL or absolute glucose > 180 mg/dL is BAD / Glycemic excursion

# Intolerance Detection Thresholds
INTOLERANCE_HISTORICAL_BASELINE_MAX = 36.0  # mg/dL (Historically well-tolerated spike)
INTOLERANCE_SPIKE_TRIGGER = 48.0            # mg/dL (Current spike triggering concern)
INTOLERANCE_DELTA_JUMP = 16.0               # mg/dL (Increase over past average)

# Default Patient Profile
DEFAULT_PATIENT = {
    "id": "pat_mahesh_001",
    "name": "Mahesh Chandran",
    "age": 52,
    "gender": "Male",
    "diabetes_type": "Type 2 Diabetes",
    "target_min": 70.0,
    "target_max": 160.0,
    "baseline_hba1c": 6.8,
}

# Default Twin Physiological Baseline Weights
# Delta Peak = w_carb * Carbs + w_fiber * Fiber + w_fat * Fat + w_protein * Protein + w_hba1c * (HbA1c - 5.5) + bias
DEFAULT_TWIN_WEIGHTS = {
    "carb_weight": 0.85,       # mg/dL rise per gram of carbohydrate
    "fiber_weight": -0.45,     # mg/dL buffering effect per gram of fiber
    "fat_weight": -0.15,       # mg/dL initial buffering / delayed emptying per gram of fat
    "protein_weight": 0.10,    # mg/dL gluconeogenesis / glucagon effect per gram of protein
    "baseline_weight": 0.05,   # carryover effect from high pre-meal glucose
    "hba1c_weight": 2.2,       # higher HbA1c shifts overall insulin resistance
    "bias": 5.0,               # basal digestive response offset
}

# Bounds for personal weight calibration (Physiological constraints)
WEIGHT_BOUNDS = {
    "carb_weight": (0.25, 2.0),
    "fiber_weight": (-1.5, -0.05),
    "fat_weight": (-0.6, 0.3),
    "protein_weight": (-0.1, 0.4),
    "baseline_weight": (0.0, 0.25),
    "hba1c_weight": (0.5, 6.0),
    "bias": (-10.0, 25.0),
}

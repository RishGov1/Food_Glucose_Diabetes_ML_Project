"""SQLite Database Manager for Diabetes Digital Twin."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from twin.config import DB_PATH, DEFAULT_PATIENT, DEFAULT_TWIN_WEIGHTS


@contextmanager
def get_db(db_path: Optional[Path] = None):
    """Context manager for thread-safe SQLite operations."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize SQLite tables for digital twin patient records, meals, and learning telemetry."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # 1. Patient Profile Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                diabetes_type TEXT NOT NULL,
                target_min REAL NOT NULL DEFAULT 70.0,
                target_max REAL NOT NULL DEFAULT 160.0,
                baseline_hba1c REAL NOT NULL DEFAULT 6.8,
                created_at TEXT NOT NULL
            )
        """)

        # 2. HbA1c Long-Term Tracking Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hba1c_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                recorded_date TEXT NOT NULL,
                hba1c_value REAL NOT NULL,
                fasting_glucose REAL,
                notes TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
        """)

        # 3. Meal and Glucose Response History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meal_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                meal_type TEXT NOT NULL,
                food_name TEXT NOT NULL,
                portion_amount REAL NOT NULL DEFAULT 1.0,
                portion_unit TEXT NOT NULL DEFAULT 'serving',
                carbs_g REAL NOT NULL,
                fiber_g REAL NOT NULL DEFAULT 0.0,
                protein_g REAL NOT NULL DEFAULT 0.0,
                fat_g REAL NOT NULL DEFAULT 0.0,
                digestion_hours REAL NOT NULL DEFAULT 1.5,
                pre_glucose REAL NOT NULL,
                glucose_15m REAL,
                glucose_30m REAL,
                glucose_45m REAL,
                glucose_60m REAL,
                glucose_120m REAL,
                peak_glucose REAL NOT NULL,
                delta_glucose REAL NOT NULL,
                verdict TEXT NOT NULL,                -- GOOD, MODERATE, BAD
                intolerance_status TEXT NOT NULL,     -- TOLERATED, EMERGING_INTOLERANT, KNOWN_INTOLERANT, UNKNOWN
                intolerance_notes TEXT,
                predicted_peak REAL NOT NULL,
                residual_error REAL NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
        """)

        # 4. Adaptive Digital Twin Personal Weights Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS twin_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                carb_weight REAL NOT NULL,
                fiber_weight REAL NOT NULL,
                fat_weight REAL NOT NULL,
                protein_weight REAL NOT NULL,
                baseline_weight REAL NOT NULL,
                hba1c_weight REAL NOT NULL,
                bias REAL NOT NULL,
                learning_step INTEGER NOT NULL DEFAULT 0,
                notes TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
        """)

        # 5. Online Learning Telemetry Events Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                meal_id INTEGER,
                old_weights_json TEXT NOT NULL,
                new_weights_json TEXT NOT NULL,
                delta_weights_json TEXT NOT NULL,
                loss REAL NOT NULL,
                trigger_reason TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (meal_id) REFERENCES meal_logs(id)
            )
        """)


def ensure_patient_profile(patient_data: Optional[Dict[str, Any]] = None, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Ensure that the default or specified patient profile exists."""
    p = patient_data or DEFAULT_PATIENT
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE id = ?", (p["id"],))
        row = cursor.fetchone()
        if not row:
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute("""
                INSERT INTO patients (id, name, age, gender, diabetes_type, target_min, target_max, baseline_hba1c, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (p["id"], p["name"], p["age"], p["gender"], p["diabetes_type"], p["target_min"], p["target_max"], p["baseline_hba1c"], now))
            
            # Seed initial weights
            cursor.execute("""
                INSERT INTO twin_weights (patient_id, updated_at, carb_weight, fiber_weight, fat_weight, protein_weight, baseline_weight, hba1c_weight, bias, learning_step, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p["id"], now,
                DEFAULT_TWIN_WEIGHTS["carb_weight"],
                DEFAULT_TWIN_WEIGHTS["fiber_weight"],
                DEFAULT_TWIN_WEIGHTS["fat_weight"],
                DEFAULT_TWIN_WEIGHTS["protein_weight"],
                DEFAULT_TWIN_WEIGHTS["baseline_weight"],
                DEFAULT_TWIN_WEIGHTS["hba1c_weight"],
                DEFAULT_TWIN_WEIGHTS["bias"],
                0, "Initial calibrated physiological baseline weights"
            ))
            return dict(p)
        return dict(row)


def get_patient(patient_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve patient record by ID."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_hba1c_history(patient_id: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get chronological HbA1c progression for the patient."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM hba1c_logs
            WHERE patient_id = ?
            ORDER BY recorded_date ASC
        """, (patient_id,))
        return [dict(r) for r in cursor.fetchall()]


def add_hba1c_log(patient_id: str, recorded_date: str, hba1c_value: float, fasting_glucose: Optional[float] = None, notes: str = "", db_path: Optional[Path] = None) -> int:
    """Record an HbA1c test result."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO hba1c_logs (patient_id, recorded_date, hba1c_value, fasting_glucose, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (patient_id, recorded_date, float(hba1c_value), fasting_glucose, notes))
        return cursor.lastrowid


def get_meal_history(patient_id: str, limit: int = 100, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get recent meals logged for the patient."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM meal_logs
            WHERE patient_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (patient_id, limit))
        return [dict(r) for r in cursor.fetchall()]


def get_food_history(patient_id: str, food_name: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all past meal logs for a specific food (or similar foods) for intolerance analysis."""
    clean_name = food_name.strip().lower()
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        # Look for exact match or substring match
        cursor.execute("""
            SELECT * FROM meal_logs
            WHERE patient_id = ? AND LOWER(food_name) LIKE ?
            ORDER BY timestamp ASC
        """, (patient_id, f"%{clean_name}%"))
        return [dict(r) for r in cursor.fetchall()]


def add_meal_log(patient_id: str, meal: Dict[str, Any], db_path: Optional[Path] = None) -> int:
    """Add a new meal reading record and return the inserted meal ID."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO meal_logs (
                patient_id, timestamp, meal_type, food_name, portion_amount, portion_unit,
                carbs_g, fiber_g, protein_g, fat_g, digestion_hours, pre_glucose,
                glucose_15m, glucose_30m, glucose_45m, glucose_60m, glucose_120m,
                peak_glucose, delta_glucose, verdict, intolerance_status, intolerance_notes,
                predicted_peak, residual_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            meal.get("timestamp", datetime.now().isoformat()),
            meal.get("meal_type", "Lunch"),
            meal["food_name"],
            float(meal.get("portion_amount", 1.0)),
            meal.get("portion_unit", "serving"),
            float(meal["carbs_g"]),
            float(meal.get("fiber_g", 0.0)),
            float(meal.get("protein_g", 0.0)),
            float(meal.get("fat_g", 0.0)),
            float(meal.get("digestion_hours", 1.5)),
            float(meal["pre_glucose"]),
            float(meal.get("glucose_15m")) if meal.get("glucose_15m") is not None else None,
            float(meal.get("glucose_30m")) if meal.get("glucose_30m") is not None else None,
            float(meal.get("glucose_45m")) if meal.get("glucose_45m") is not None else None,
            float(meal.get("glucose_60m")) if meal.get("glucose_60m") is not None else None,
            float(meal.get("glucose_120m")) if meal.get("glucose_120m") is not None else None,
            float(meal["peak_glucose"]),
            float(meal["delta_glucose"]),
            meal["verdict"],
            meal.get("intolerance_status", "TOLERATED"),
            meal.get("intolerance_notes", ""),
            float(meal.get("predicted_peak", meal["peak_glucose"])),
            float(meal.get("residual_error", 0.0))
        ))
        return cursor.lastrowid


def get_latest_weights(patient_id: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Retrieve current adaptive weights for patient digital twin."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM twin_weights
            WHERE patient_id = ?
            ORDER BY id DESC
            LIMIT 1
        """, (patient_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        # Fallback to default
        return {**DEFAULT_TWIN_WEIGHTS, "learning_step": 0, "patient_id": patient_id}


def save_weights(patient_id: str, weights: Dict[str, float], step_info: Dict[str, Any], db_path: Optional[Path] = None) -> int:
    """Save newly learned weights and log telemetry event."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        new_step = int(step_info.get("learning_step", 1))

        cursor.execute("""
            INSERT INTO twin_weights (
                patient_id, updated_at, carb_weight, fiber_weight, fat_weight,
                protein_weight, baseline_weight, hba1c_weight, bias, learning_step, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id, now,
            float(weights["carb_weight"]),
            float(weights["fiber_weight"]),
            float(weights["fat_weight"]),
            float(weights["protein_weight"]),
            float(weights["baseline_weight"]),
            float(weights["hba1c_weight"]),
            float(weights["bias"]),
            new_step,
            step_info.get("notes", "Online gradient adaptive weight update")
        ))
        weight_id = cursor.lastrowid

        # Record learning event
        if "old_weights" in step_info:
            cursor.execute("""
                INSERT INTO learning_events (
                    patient_id, timestamp, meal_id, old_weights_json, new_weights_json, delta_weights_json, loss, trigger_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id, now,
                step_info.get("meal_id"),
                json.dumps(step_info.get("old_weights", {})),
                json.dumps(weights),
                json.dumps(step_info.get("delta_weights", {})),
                float(step_info.get("loss", 0.0)),
                step_info.get("trigger_reason", "Observed meal postprandial residual update")
            ))

        return weight_id


def get_learning_history(patient_id: str, limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve history of weight adjustments and learning telemetry."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM learning_events
            WHERE patient_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (patient_id, limit))
        return [dict(r) for r in cursor.fetchall()]


def get_patient_summary_stats(patient_id: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Calculate key clinical and twin metrics for dashboard overview."""
    with get_db(db_db := db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM meal_logs WHERE patient_id = ?", (patient_id,))
        meals = [dict(r) for r in cursor.fetchall()]

        cursor.execute("SELECT * FROM hba1c_logs WHERE patient_id = ? ORDER BY recorded_date DESC LIMIT 1", (patient_id,))
        latest_hba1c_row = cursor.fetchone()
        latest_hba1c = latest_hba1c_row["hba1c_value"] if latest_hba1c_row else DEFAULT_PATIENT["baseline_hba1c"]

        total_meals = len(meals)
        if total_meals == 0:
            return {
                "total_meals": 0,
                "good_meals": 0,
                "moderate_meals": 0,
                "bad_meals": 0,
                "tir_percent": 0.0,
                "avg_delta": 0.0,
                "latest_hba1c": latest_hba1c,
                "intolerant_foods_count": 0,
            }

        good = sum(1 for m in meals if m["verdict"] == "GOOD")
        moderate = sum(1 for m in meals if m["verdict"] == "MODERATE")
        bad = sum(1 for m in meals if m["verdict"] == "BAD")
        in_range = sum(1 for m in meals if (m.get("glucose_120m") or m["peak_glucose"]) <= 160.0 and (m.get("glucose_120m") or m["peak_glucose"]) >= 70.0)
        tir = (in_range / total_meals) * 100.0
        avg_delta = sum(m["delta_glucose"] for m in meals) / total_meals

        # Count foods currently flagged with intolerance
        intolerant = set(m["food_name"] for m in meals if m.get("intolerance_status") in ("EMERGING_INTOLERANT", "KNOWN_INTOLERANT"))

        return {
            "total_meals": total_meals,
            "good_meals": good,
            "moderate_meals": moderate,
            "bad_meals": bad,
            "tir_percent": round(tir, 1),
            "avg_delta": round(avg_delta, 1),
            "latest_hba1c": round(latest_hba1c, 1),
            "intolerant_foods_count": len(intolerant),
            "intolerant_foods": sorted(list(intolerant)),
        }

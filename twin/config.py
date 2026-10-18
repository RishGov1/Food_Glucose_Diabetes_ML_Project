from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FEATURES = ["baseline", "carb_g", "protein_g", "fat_g", "fibre_g", "sugar_g", "activity_min", "sleep_hours"]
TARGETS = ["g60", "g120"]
NUTRIENTS = ["energy_kcal", "carb_g", "protein_g", "fat_g", "fibre_g", "sugar_g"]
SOURCE_URL = "https://www.anuvaad.org.in/wp-content/uploads/2020/07/ANUVAAD_INDB_2024.xlsx"
SOURCE_PAGE = "https://www.anuvaad.org.in/indian-nutrient-databank/"
SEED = 20261018

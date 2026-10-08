from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, ensure_parent
from src.d1namo_loader import load_d1namo

if __name__ == "__main__":
    cfg = load_config()
    df = load_d1namo(
        cfg["paths"]["d1namo_root"],
        meal_gap_minutes=cfg["project"]["meal_gap_minutes"],
        tolerance_minutes=cfg["project"]["glucose_tolerance_minutes"],
    )
    ensure_parent(cfg["paths"]["samples_csv"])
    df.to_csv(cfg["paths"]["samples_csv"], index=False)

    print(f"Created {len(df)} meal samples.")
    print("Subjects:", df["subject_id"].nunique())
    print("Class distribution:")
    print(df.groupby("label")["subject_id"].nunique())
    print("\nSix-point availability:")
    for m in [0, 15, 40, 45, 60, 120]:
        col = f"g_{m}"
        print(f"  {col}: {df[col].notna().mean()*100:.1f}%")

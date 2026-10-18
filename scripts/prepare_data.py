"""Preserve source files; normalize food provenance and synthesize labelled scenarios."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import requests
from twin.config import DATA, FEATURES, NUTRIENTS, PROCESSED, REPORTS, SEED, SOURCE_PAGE, SOURCE_URL


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def category(name, supplied):
    if pd.notna(supplied):
        return str(supplied).strip()
    name = name.casefold()
    for group, words in [
        ("Sweets & desserts", ["sweet", "halwa", "payasam", "kheer", "ladoo", "laddu", "burfi", "barfi", "jamun", "jalebi", "cake", "pudding"]),
        ("Drinks", ["tea", "coffee", "juice", "lassi", "sherbet", "drink", "milkshake"]),
        ("Rice & grains", ["rice", "biryani", "pulao", "khichdi", "pongal", "bath", "poha"]),
        ("Breads & breakfast", ["roti", "chapati", "paratha", "dosa", "idli", "bread", "upma", "uttapam", "puri", "appam"]),
        ("Pulses & legumes", ["dal", "chana", "chole", "rajma", "sambar", "lentil", "bean", "sprout"]),
        ("Vegetables & sides", ["vegetable", "sabzi", "cabbage", "carrot", "spinach", "bhindi", "brinjal", "gourd", "salad"]),
        ("Eggs, fish & meat", ["chicken", "mutton", "fish", "egg", "prawn", "meat"]),
        ("Dairy", ["paneer", "curd", "yogurt", "milk", "cheese"]),
    ]:
        if any(w in name for w in words):
            return group
    return "Other recipes"


def build_catalog():
    source = DATA / "sources/ANUVAAD_INDB_2024.xlsx"
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        response = requests.get(SOURCE_URL, timeout=90)
        response.raise_for_status()
        source.write_bytes(response.content)
    official = pd.read_excel(source, sheet_name="INDB")
    original = pd.read_excel(DATA / "food/indian_food.xlsx")
    records, rejected = [], []
    for origin, frame in [("INDB 2024", official), ("Original workbook", original)]:
        for _, row in frame.iterrows():
            name = str(row.food_name).strip()
            code = str(row.food_code)
            primary = str(row.primarysource)
            source_kind = ("estimated recipe" if primary in {"asc_manual", "bfp_manual", "open_source_recipes"}
                           else "manual estimate" if primary == "manual_estimate" else "composition table")
            nutrients = {k: float(row.get("freesugar_g" if k == "sugar_g" else k, np.nan)) for k in NUTRIENTS}
            issues = []
            if not all(np.isfinite(v) and v >= 0 for v in nutrients.values()):
                issues.append("missing, nonfinite or negative core nutrient")
            if any(nutrients[k] > 100.5 for k in NUTRIENTS if k != "energy_kcal"):
                issues.append("nutrient exceeds 100 g per 100 g")
            if nutrients["energy_kcal"] > 950 or nutrients["sugar_g"] > nutrients["carb_g"] + 1:
                issues.append("implausible energy or sugar exceeds carbohydrate")
            if nutrients["carb_g"] + nutrients["protein_g"] + nutrients["fat_g"] > 105:
                issues.append("macronutrient mass exceeds edible mass")
            if issues:
                rejected.append({"origin": origin, "food_code": code, "name": name, "issues": "; ".join(issues)})
                continue
            ratios = []
            for k in ["carb_g", "protein_g", "fat_g", "energy_kcal"]:
                serving = row.get("unit_serving_" + k, np.nan)
                if pd.notna(serving) and nutrients[k] > 0.1:
                    ratios.append(float(serving) / nutrients[k] * 100)
            grams = float(np.median(ratios)) if ratios else 100.0
            consistent = ratios and max(abs(np.array(ratios) - grams)) < max(2, grams * .05)
            grams_valid = consistent and 5 <= grams <= 750
            grams = round(grams, 1) if grams_valid else 100.0
            unit = str(row.get("servings_unit", ""))
            unit = unit if unit and unit != "nan" and grams_valid else "100 g reference"
            key = normalize(name)
            records.append({
                "food_id": "food_" + hashlib.sha256(key.encode()).hexdigest()[:14], "name": name,
                "family_id": key, "source_code": code, "source": origin, "primary_source": primary,
                "source_url": SOURCE_PAGE if origin == "INDB 2024" else "data/food/indian_food.xlsx",
                "evidence": source_kind, "category": category(name, row.get("food_group_nin", np.nan)),
                "is_recipe": source_kind != "composition table",
                "meal_eligible": source_kind != "composition table" or row.get("food_group_nin", "") == "Fruits",
                "serving_g": grams, "serving_unit": unit,
                "serving_basis": "inferred from published nutrient ratios" if grams_valid else "100 g reference; weigh your portion",
                **nutrients,
            })
    all_rows = pd.DataFrame(records)
    duplicates = all_rows[all_rows.duplicated("family_id", keep=False)].copy()
    catalog = all_rows.drop_duplicates("family_id", keep="first").reset_index(drop=True)
    catalog.to_csv(PROCESSED / "foods.csv", index=False)
    duplicates.to_csv(PROCESSED / "duplicate_food_audit.csv", index=False)
    pd.DataFrame(rejected, columns=["origin", "food_code", "name", "issues"]).to_csv(PROCESSED / "quarantined_foods.csv", index=False)
    # These are explicitly portion scenarios, never counted as new foods or patients.
    portions = pd.concat([catalog.assign(portion_multiplier=m, portion_g=catalog.serving_g * m)
                          for m in [.5, .75, 1, 1.5, 2]], ignore_index=True)
    for col in NUTRIENTS:
        portions[col] = portions[col] * portions.portion_g / 100
    portions["provenance"] = "calculated portion; not measured meal response"
    portions.to_csv(PROCESSED / "food_portions.csv", index=False)
    report = {
        "source_url": SOURCE_URL, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "retrieved_date": "2026-10-18", "original_rows": len(original), "original_duplicate_names": int(original.food_name.duplicated().sum()),
        "source_rows": len(official), "source_names_absent_from_original": len(set(map(normalize, official.food_name)) - set(map(normalize, original.food_name))),
        "catalog_unique_foods": len(catalog), "recipe_foods": int(catalog.is_recipe.sum()),
        "ingredient_foods": int((~catalog.is_recipe).sum()), "meal_eligible_foods":int(catalog.meal_eligible.sum()),
        "quarantined_rows": len(rejected),
        "deduplicated_rows": len(all_rows) - len(catalog), "calculated_portion_rows": len(portions),
        "source_counts": catalog.primary_source.value_counts().to_dict(), "category_counts": catalog.category.value_counts().to_dict(),
        "limitations": ["No glycemic index values inferred from names.", "Recipe nutrients are estimates, not laboratory assays.",
                        "Source carbohydrate definitions differ; fibre is not subtracted again.",
                        "Name deduplication does not establish semantic equivalence of differently named dishes.",
                        "More foods do not remove patient, geography, treatment or outcome bias."]}
    return catalog, report


def synthesize(catalog, count):
    rng = np.random.default_rng(SEED)
    recipes = catalog[catalog.meal_eligible & (catalog.energy_kcal > 5)].copy()
    # Assign canonical food families to partitions BEFORE sampling any portions or patients.
    recipes = recipes.iloc[rng.permutation(len(recipes))].reset_index(drop=True)
    cuts = [0, int(len(recipes)*.7), int(len(recipes)*.8), int(len(recipes)*.9), len(recipes)]
    names, fractions = ["train", "validation", "calibration", "test"], [.7, .1, .1, .1]
    frames = []
    remaining = count
    for split_no, (name, fraction) in enumerate(zip(names, fractions)):
        n = int(count*fraction) if split_no < 3 else remaining
        remaining -= n
        food = recipes.iloc[cuts[split_no]:cuts[split_no+1]]
        idx = rng.integers(0, len(food), n)
        idx2 = rng.integers(0, len(food), n)
        a, b = food.iloc[idx].reset_index(drop=True), food.iloc[idx2].reset_index(drop=True)
        qa = rng.uniform(.4, 1.8, n) * a.serving_g.to_numpy() / 100
        qb = rng.uniform(.2, .8, n) * b.serving_g.to_numpy() / 100 * (rng.random(n) < .3)
        macro = a[NUTRIENTS].to_numpy()*qa[:,None] + b[NUTRIENTS].to_numpy()*qb[:,None]
        # Cap whole meal quantities together to preserve the food's composition.
        limits = np.array([1500, 180, 100, 100, 50, 80])
        scale = np.minimum(1, np.min(limits / np.maximum(macro, .001), axis=1))
        macro *= scale[:,None]
        people = max(30, n // 50)
        patient = rng.integers(0, people, n)
        hidden_sensitivity = rng.lognormal(0, .22, people)[patient]
        hidden_recovery = rng.uniform(.45, .95, people)[patient]
        baseline = rng.uniform(80, 240, n)
        activity = rng.choice([0, 10, 20, 30, 45, 60], n)
        sleep = rng.uniform(4, 10, n)
        _, carbs, protein, fat, fibre, sugar = macro.T
        # Assumed simulator, NOT clinical physiology or measured GI. See MODEL_REPORT.md.
        load = carbs / (1 + .025*fibre + .008*protein + .006*fat)
        response = (1.05*load + .22*sugar) * hidden_sensitivity * (1 + .035*(7-sleep)) * np.exp(-.003*activity)
        shared_noise = rng.normal(0, 7, n)
        g60 = baseline + response + shared_noise + rng.normal(0, 5, n)
        g120 = baseline + hidden_recovery*response + .10*fat - .12*(baseline-110) + shared_noise + rng.normal(0, 7, n)
        frame = pd.DataFrame(macro, columns=NUTRIENTS)
        frame["baseline"], frame["activity_min"], frame["sleep_hours"] = baseline, activity, sleep
        frame["g60"], frame["g120"] = np.clip(g60, 40, 500), np.clip(g120, 40, 500)
        frame["person_id"] = [f"synthetic_{name}_{p:05d}" for p in patient]
        frame["food_id"] = a.food_id
        frame["second_food_id"] = np.where(qb > 0, b.food_id, "")
        frame["split"] = name
        frame["provenance"] = "synthetic_simulator_v1_NOT_PATIENT_DATA"
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.insert(0, "scenario_id", [f"SIM{i:07d}" for i in range(len(result))])
    result.to_csv(PROCESSED / "synthetic_meals.csv.gz", index=False, float_format="%.5f", compression={"method":"gzip","mtime":0})
    return {"synthetic_rows": len(result), "split_rows": result.split.value_counts().to_dict(),
            "seed": SEED, "synthetic_patients": int(result.person_id.nunique()),
            "simulation_features": FEATURES, "synthetic_target_claim": "Simulator imitation only. NOT clinical accuracy."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100000)
    args = parser.parse_args()
    if not 1000 <= args.rows <= 1000000:
        parser.error("--rows must be between 1000 and 1000000")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    catalog, report = build_catalog()
    report.update(synthesize(catalog, args.rows))
    (REPORTS / "data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ["category_counts", "source_counts"]}, indent=2))


if __name__ == "__main__":
    main()

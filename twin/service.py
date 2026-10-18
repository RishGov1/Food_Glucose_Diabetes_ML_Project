"""Nutrition, contextual estimates, evidence-aware food lists and safety gating."""
import math
import numpy as np
import pandas as pd
from twin.config import FEATURES, NUTRIENTS, PROCESSED
from twin.learning import apply_personal, key_for


def load_catalog(custom=None):
    base=pd.read_csv(PROCESSED/"foods.csv").fillna({"source_url":"","serving_unit":"100 g reference"})
    if custom:
        base=pd.concat([base,pd.DataFrame(custom)],ignore_index=True)
    return base


def glucose_to_mg(value, unit):
    number=float(value)
    if unit not in ["mg/dL","mmol/L"] or not math.isfinite(number):
        raise ValueError("Select valid glucose units and enter a finite value.")
    number=number*18.0182 if unit=="mmol/L" else number
    if not 20<=number<=600:
        raise ValueError("Glucose must be between 20 and 600 mg/dL (approximately 1.11 to 33.29 mmol/L).")
    return number


def safety_message(glucose):
    if glucose<54:
        return "Very low glucose. Follow your prescribed low-glucose plan now. Get emergency help for confusion, fainting, seizures or inability to swallow. Food ranking is paused."
    if glucose<70:
        return "Low glucose. Follow your prescribed low-glucose treatment plan and recheck as instructed. Food ranking is paused."
    if glucose>250:
        return "High glucose. Follow your prescribed high-glucose plan and contact your diabetes care team. Seek urgent care for vomiting, abdominal pain, deep breathing, confusion or positive ketones. Food ranking is paused."
    return ""


def meal_totals(catalog, selected):
    if not selected:
        raise ValueError("Choose at least one food.")
    if len(selected)>12 or len({i[0] for i in selected})!=len(selected):
        raise ValueError("Use up to 12 distinct foods; combine repeated portions into one weight.")
    items=[]
    totals={k:0.0 for k in NUTRIENTS}
    lookup=catalog.set_index("food_id")
    for food_id,grams in selected:
        grams=float(grams)
        if not math.isfinite(grams) or not 5<=grams<=1000:
            raise ValueError("Each portion must be between 5 and 1,000 g.")
        if food_id not in lookup.index:
            raise ValueError("A selected food is unavailable in this profile.")
        row=lookup.loc[food_id]
        nutrients={k:float(row[k])*grams/100 for k in NUTRIENTS}
        if not all(math.isfinite(v) and v>=0 for v in nutrients.values()):
            raise ValueError("Incomplete food nutrition cannot be used for prediction.")
        item={"food_id":food_id,"name":str(row["name"]),"grams":grams,"evidence":str(row.evidence),
              "is_recipe":bool(row.is_recipe),"meal_eligible":bool(row.meal_eligible),"source":str(row.source),**nutrients}
        items.append(item)
        totals={k:totals[k]+nutrients[k] for k in NUTRIENTS}
    return totals,items


def validate_context(context):
    ranges={"baseline":(80,240),"carb_g":(0,180),"protein_g":(0,100),"fat_g":(0,100),
            "fibre_g":(0,50),"sugar_g":(0,80),"activity_min":(0,60),"sleep_hours":(4,10)}
    for key,(low,high) in ranges.items():
        value=float(context[key])
        if not math.isfinite(value) or not low<=value<=high:
            return f"{key.replace('_',' ')} is outside the model's development range ({low}-{high}). No food verdict is available."
    return ""


def measured_evidence(rows, items, context, low, high):
    compatible=[]
    for r in rows:
        if r.get("observed120") is None or r.get("provenance")!="measured" or r.get("excluded_reason") or r.get("confounded"):
            continue
        if key_for(r["items"])!=key_for(items) or abs(r["context"]["baseline"]-context["baseline"])>30:
            continue
        # Require each component to have a similar portion, not merely the same total mass.
        portions={i["food_id"]:i["grams"] for i in r["items"]}
        if not all(.75<=portions[i["food_id"]]/i["grams"]<=1.25 for i in items):
            continue
        compatible.append(r)
    n=len(compatible)
    above=sum(r["observed120"]>=high for r in compatible)
    below=sum(r["observed120"]<low for r in compatible)
    inside=n-above-below
    return {"n":n,"in_target":inside,"above":above,"below":below,
            "mean_120":float(np.mean([r["observed120"] for r in compatible])) if n else None,
            "note":"Same meal components, each portion within 25%, premeal glucose within 30 mg/dL. Observational association, not causation."}


def classify(mean, lower, upper, evidence, low, high):
    if evidence["below"]:
        return "Review with care team", "A comparable measured meal was below your lower target."
    if evidence["n"]>=3:
        if evidence["above"] / evidence["n"] >= .5:
            return "Limit / discuss", "At least half of comparable measured meals exceeded your upper target."
        if evidence["in_target"] / evidence["n"]>=.8 and upper<high and lower>=low:
            return "Likely suitable", "At least 80% of comparable meals were in target, and the illustrative range is within target."
    if lower<low:
        return "Review with care team", "The illustrative range crosses your lower target."
    if mean>=high:
        return "Review portion", "The illustrative two-hour estimate exceeds your upper target; confirm with measured outcomes."
    if upper>=high:
        return "Uncertain", "The illustrative range crosses your upper target. More measured outcomes are needed."
    return "Promising / unconfirmed", "The illustrative range is within target, but repeated personal measurements are still needed."


def predict_meal(artifact, personal, context, items, rows, profile):
    problem=safety_message(context["baseline"]) or validate_context(context)
    if problem:
        return {"blocked":True,"reason":problem}
    if any(not i["meal_eligible"] for i in items):
        return {"blocked":True,"reason":"This selection includes ingredient composition measured in its listed state. Add a cooked recipe or a labelled ready-to-eat food for a meal estimate; cooking and edible weight matter."}
    x=np.array([[context[k] for k in FEATURES]])
    base=np.asarray(artifact["model"].predict(x)[0],dtype=float)
    adjustment=apply_personal(personal,context,items)
    mean=base+adjustment
    width=np.maximum(35,np.asarray(artifact["q90"],dtype=float))+np.abs(adjustment)*.25
    # User-entered/manual nutrition has additional uncertainty, never greater confidence.
    if any("unverified" in i["evidence"] or "manual estimate" in i["evidence"] for i in items):
        width+=10
    lower,upper=np.maximum(20,mean-width),np.minimum(600,mean+width)
    evidence=measured_evidence(rows,items,context,profile["target_low"],profile["target_high"])
    status,reason=classify(mean[1],lower[1],upper[1],evidence,profile["target_low"],profile["target_high"])
    return {"blocked":False,"base":base.tolist(),"mean":mean.tolist(),"lower":lower.tolist(),"upper":upper.tolist(),
            "adjustment":adjustment.tolist(),"evidence":evidence,"status":status,"reason":reason,
            "version":artifact["report"]["version"],"personal_version":personal["version"]}

"""Measured-only sequential residual learning with a chronological activation gate."""
import hashlib
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from twin.config import FEATURES


def eligible(rows, version):
    # A prediction can never be its own training label; historical demo databases are not imported.
    return [r for r in rows if r.get("provenance")=="measured" and r.get("observed120") is not None
            and not r.get("excluded_reason") and not r.get("confounded") and r.get("model_version")==version]


def key_for(items):
    return "|".join(sorted(item["food_id"] for item in items))


def residual_offset(rows, items, context, target):
    if not rows:
        return 0.0
    residual = np.array([r[f"observed{target}"]-r[f"base{target}"] for r in rows])
    offset = np.sum(np.clip(residual,-80,80))/(len(rows)+8)
    matching = [r for r in rows if key_for(r["items"])==key_for(items)
                and abs(r["context"]["baseline"]-context["baseline"])<=30
                and .75 <= sum(x["grams"] for x in r["items"])/sum(x["grams"] for x in items) <= 1.25]
    if matching:
        local=sum(np.clip(r[f"observed{target}"]-r[f"base{target}"]-offset,-80,80) for r in matching)
        offset+=local/(len(matching)+5)
    return float(np.clip(offset,-40,40))


def fit_ridge(rows, target):
    x=np.array([[r["context"][k] for k in FEATURES] for r in rows])
    y=np.array([r[f"observed{target}"]-r[f"base{target}"] for r in rows])
    return make_pipeline(StandardScaler(),Ridge(alpha=40)).fit(x,y)


def fit_personal(rows, version):
    clean=eligible(rows,version)
    fingerprint=hashlib.sha256(json.dumps([(r["id"],r["observed120"],r["observed60"]) for r in clean]).encode()).hexdigest()[:12]
    result={"version":f"personal-{fingerprint}","eligible_meals":len(clean),"targets":{},"source":"measured outcomes only"}
    for target in [60,120]:
        records=[r for r in clean if r.get(f"observed{target}") is not None]
        # Outcome availability must precede validation meal time, even for backfilled records.
        records=sorted(records,key=lambda r:r["eaten_at"])
        entry={"mode":"collecting","n":len(records),"active":False,"rows":records}
        if len(records)<12:
            entry["reason"]="At least 12 eligible measured meals are needed for an activation check."
            result["targets"][target]=entry
            continue
        nval=max(4,min(12,len(records)//3))
        errors={"base":[],"offset":[],"ridge":[]}
        for i in range(len(records)-nval,len(records)):
            row=records[i]
            past=[r for r in records[:i] if (r.get(f"measured_at{target}") or r["measured_at"])<=row["eaten_at"]]
            if len(past)<8:
                continue
            truth=row[f"observed{target}"]
            base=row[f"base{target}"]
            errors["base"].append(abs(truth-base))
            offset=residual_offset(past,row["items"],row["context"],target)
            errors["offset"].append(abs(truth-base-offset))
            model=fit_ridge(past,target)
            correction=float(np.clip(model.predict([[row["context"][k] for k in FEATURES]])[0],-40,40))
            errors["ridge"].append(abs(truth-base-correction))
        if len(errors["base"])<4:
            entry["reason"]="Need at least four later meals whose earlier outcomes were already recorded."
        else:
            mae={k:float(np.mean(v)) for k,v in errors.items()}
            selected=min(["offset","ridge"],key=lambda k:mae[k])
            active=mae[selected] <= .95*mae["base"] and mae["base"]-mae[selected]>=1
            entry.update({"validation_mae":mae,"validation_meals":len(errors["base"]),"active":active,
                          "mode":selected if active else "base retained",
                          "reason":"Improved later-meal MAE by at least 5% and 1 mg/dL." if active else "Personal update did not improve the base estimate enough."})
            if active and selected=="ridge":
                entry["model"]=fit_ridge(records,target)
        result["targets"][target]=entry
    return result


def apply_personal(state, context, items):
    changes=[]
    for target in [60,120]:
        entry=state["targets"][target]
        if not entry["active"]:
            changes.append(0.)
        elif entry["mode"]=="offset":
            changes.append(residual_offset(entry["rows"],items,context,target))
        else:
            changes.append(float(np.clip(entry["model"].predict([[context[k] for k in FEATURES]])[0],-40,40)))
    return np.array(changes)


def learning_summary(state):
    return {"version":state["version"],"eligible_meals":state["eligible_meals"],"source":state["source"],
            "targets":{str(k):{a:b for a,b in v.items() if a not in ["rows","model"]} for k,v in state["targets"].items()}}

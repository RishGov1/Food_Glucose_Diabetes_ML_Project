"""Independent observed-glucose benchmark; never joins unrelated food/nutrition rows."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from twin.config import DATA, MODELS, PROCESSED, REPORTS, SEED
from twin.estimators import Persistence, ResidualModel
from scripts.train_models import scores


def nearest(times, values, target, tolerance=8, cutoff=None):
    valid = np.ones(len(times), dtype=bool) if cutoff is None else times <= cutoff
    distances = np.abs((times-target)/np.timedelta64(1,"m"))
    distances[~valid] = np.inf
    k = int(np.argmin(distances))
    return float(values[k]) if distances[k] <= tolerance else np.nan


def extract():
    rows, audited = [], []
    for file in sorted((DATA / "raw/d1namo").rglob("glucose.csv")):
        if "diabetes_subset" not in str(file):
            continue  # Only the explicitly identified Type 1 cohort; no label inference.
        glucose = pd.read_csv(file)
        food = pd.read_csv(file.parent / "food.csv")
        stamps = pd.to_datetime(glucose.date.astype(str)+" "+glucose.time.astype(str), errors="coerce")
        # Original D1NAMO concentration column is mmol/L; never infer units from a patient's values.
        values = pd.to_numeric(glucose.glucose, errors="coerce") * 18.0182
        frame = pd.DataFrame({"time": stamps, "value":values}).dropna().sort_values("time").groupby("time",as_index=False).value.mean()
        frame = frame[frame.value.between(20,600)]
        if frame.empty:
            continue
        ft = food.datetime.astype(str).str.replace(r"^(\d{4}):(\d{2}):(\d{2})",r"\1-\2-\3",regex=True)
        meal_times = pd.to_datetime(ft, errors="coerce", format="mixed").dropna().drop_duplicates().sort_values()
        groups = (meal_times.diff().dt.total_seconds().fillna(1e6) > 30*60).cumsum()
        meal_times = meal_times.groupby(groups).min().tolist()
        times, gv = frame.time.to_numpy(), frame.value.to_numpy()
        eligible = 0
        for i, meal in enumerate(meal_times):
            if i+1 < len(meal_times) and meal_times[i+1]-meal <= pd.Timedelta(minutes=128):
                continue  # Avoid another recorded meal contaminating the outcome window.
            entry = {"participant": "t1d_"+file.parent.name, "meal_time": str(meal)}
            for offset in [0,15,40,45,60,120]:
                target = np.datetime64(meal+pd.Timedelta(minutes=offset))
                cutoff = np.datetime64(meal+pd.Timedelta(minutes=45)) if offset <= 45 else None
                entry[f"g{offset}"] = nearest(times,gv,target,cutoff=cutoff)
            if all(np.isfinite(entry[f"g{m}"]) for m in [0,15,40,45,60,120]):
                rows.append(entry)
                eligible += 1
        audited.append({"participant": "t1d_"+file.parent.name,"meal_events":len(meal_times),"complete_unconfounded_windows":eligible})
    return pd.DataFrame(rows), audited


def main():
    data, audit = extract()
    data.to_csv(PROCESSED / "observed_glucose_windows.csv",index=False)
    features = ["g0","g15","g40","g45"]
    x,y,groups = data[features].to_numpy(),data[["g60","g120"]].to_numpy(),data.participant.to_numpy()
    train,test = next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=SEED).split(x,y,groups))
    candidates = {"Persistence":Persistence(column=3),
                  "Ridge":ResidualModel(make_pipeline(StandardScaler(),Ridge(alpha=20)),column=3),
                  "Extra Trees":ResidualModel(ExtraTreesRegressor(n_estimators=240,min_samples_leaf=3,n_jobs=4,random_state=SEED),column=3),
                  "Random Forest":ResidualModel(RandomForestRegressor(n_estimators=240,min_samples_leaf=3,n_jobs=4,random_state=SEED),column=3)}
    folds = GroupKFold(n_splits=min(4,len(np.unique(groups[train]))))
    table, oof = [], {}
    for name, model in candidates.items():
        pred = np.zeros_like(y[train])
        for fit_idx,val_idx in folds.split(x[train],y[train],groups[train]):
            fitted=clone(model).fit(x[train][fit_idx],y[train][fit_idx])
            pred[val_idx]=fitted.predict(x[train][val_idx])
        table.append({"model":name,**scores(y[train],pred)})
        oof[name]=pred
    selected = min(table,key=lambda r:r["mae"])["model"]
    model = clone(candidates[selected]).fit(x[train],y[train])
    q = np.quantile(np.abs(y[train]-oof[selected]),.9,axis=0,method="higher")
    report = {"cohort":"D1NAMO Type 1; observed glucose, no matched Indian meal composition",
              "rows":len(data),"participants":int(data.participant.nunique()),"extraction":audit,
              "features":features,"landmark_minutes":45,"tolerance_minutes":8,"input_unit":"mmol/L converted with factor 18.0182",
              "training_participants":sorted(set(groups[train])),"test_participants":sorted(set(groups[test])),
              "training_rows":len(train),"test_rows":len(test),"cv_leaderboard":table,"selected_model":selected,
              "test":scores(y[test],model.predict(x[test])),"test_persistence":scores(y[test],Persistence(column=3).predict(x[test])),
              "oof_90_residual":q.tolist(),"clinical_accuracy":"Small retrospective benchmark only; not clinical validation.",
              "scope":"45-minute glucose follow-up only; cannot validate food suitability or the synthetic meal model."}
    # Keep holdout participants excluded from the distributed fitted artifact.
    joblib.dump({"model":model,"q90":q,"report":report},MODELS/"observed_forecast.joblib",compress=3)
    (REPORTS/"observed_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()

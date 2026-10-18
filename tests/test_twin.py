import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.train_models import assert_split_integrity
from twin.config import DATA, FEATURES, MODELS, PROCESSED, REPORTS
from twin.learning import fit_personal, apply_personal, eligible
from twin.service import glucose_to_mg, load_catalog, meal_totals, predict_meal, measured_evidence, classify
from twin.storage import Store


@pytest.fixture(scope="module")
def artifact():
    return joblib.load(MODELS/"meal_prior.joblib")


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path/"test.sqlite3")


@pytest.fixture
def meal(artifact):
    catalog=load_catalog()
    food=catalog[catalog.is_recipe & catalog.name.str.contains("idli",case=False)].iloc[0]
    total,items=meal_totals(catalog,[(food.food_id,100)])
    context={"baseline":110.,"activity_min":0.,"sleep_hours":7.,**{k:total[k] for k in FEATURES if k in total}}
    profile={"target_low":70,"target_high":180}
    personal=fit_personal([],artifact["report"]["version"])
    pred=predict_meal(artifact,personal,context,items,[],profile)
    return context,items,pred


def test_original_data_is_bitwise_unchanged():
    manifest=json.loads((REPORTS/"original_data_manifest.json").read_text())
    assert len(manifest)>400
    for relative,digest in manifest.items():
        assert hashlib.sha256((DATA/relative).read_bytes()).hexdigest()==digest, relative


def test_synthetic_split_and_provenance():
    df=pd.read_csv(PROCESSED/"synthetic_meals.csv.gz")
    assert len(df)==100000
    assert df.scenario_id.is_unique
    assert df.provenance.str.startswith("synthetic_").all()
    assert_split_integrity(df)
    assert np.isfinite(df[FEATURES+["g60","g120"]]).all().all()


def test_food_quality_and_provenance():
    df=load_catalog()
    assert len(df)>2100
    assert df.food_id.is_unique
    assert df.family_id.is_unique
    assert (df.carb_g<=100.5).all()
    assert df.source.notna().all()
    assert df.evidence.notna().all()
    assert len(pd.read_csv(PROCESSED/"quarantined_foods.csv"))==7


def test_meal_scaling_and_ingredients(artifact):
    df=load_catalog()
    food=df[df.is_recipe].iloc[0]
    a,_=meal_totals(df,[(food.food_id,100)])
    b,_=meal_totals(df,[(food.food_id,200)])
    assert b["carb_g"]==pytest.approx(2*a["carb_g"])
    ingredient=df[~df.meal_eligible].iloc[0]
    total,items=meal_totals(df,[(ingredient.food_id,50)])
    ctx={"baseline":110,"activity_min":0,"sleep_hours":7,**{k:total[k] for k in FEATURES if k in total}}
    pred=predict_meal(artifact,fit_personal([],artifact["report"]["version"]),ctx,items,[],{"target_low":70,"target_high":180})
    assert pred["blocked"]


@pytest.mark.parametrize("value",[float("nan"),float("inf"),-1,700])
def test_invalid_glucose(value):
    with pytest.raises(ValueError):
        glucose_to_mg(value,"mg/dL")


def test_explicit_units():
    assert glucose_to_mg(6,"mmol/L")==pytest.approx(108.1092)
    assert glucose_to_mg(25,"mg/dL")==25  # Never auto-convert a dangerously low reading.


@pytest.mark.parametrize("glucose",[40,65,75,245,300])
def test_out_of_range_abstention(artifact,meal,glucose):
    context,items,_=meal
    context["baseline"]=glucose
    pred=predict_meal(artifact,fit_personal([],artifact["report"]["version"]),context,items,[],{"target_low":70,"target_high":180})
    assert pred["blocked"]


def test_cold_start_cannot_be_allowed(meal):
    _,_,prediction=meal
    assert prediction["status"]!="Likely suitable"
    assert prediction["evidence"]["n"]==0
    assert prediction["upper"][1]-prediction["lower"][1]>=70


def test_duplicate_and_profile_isolation(store,meal):
    a=store.create_profile("A")
    b=store.create_profile("B")
    ctx,items,pred=meal
    time=(datetime.now(timezone.utc)-timedelta(days=1)).isoformat()
    ident=store.log_meal(a,time,items,ctx,pred)
    with pytest.raises(ValueError,match="already logged"):
        store.log_meal(a,time,items,ctx,pred)
    with pytest.raises(ValueError,match="not found"):
        store.record_outcome(b,ident,150,confirmed=True)
    assert not store.meals(b)
    with pytest.raises(ValueError,match="Confirm"):
        store.record_outcome(a,ident,150)
    store.record_outcome(a,ident,150,confirmed=True)
    with pytest.raises(ValueError,match="already has"):
        store.record_outcome(a,ident,150,confirmed=True)
    assert len(store.meals(a))==1


def test_future_and_bad_timing(store,meal):
    pid=store.create_profile("Timing")
    ctx,items,pred=meal
    with pytest.raises(ValueError,match="already started"):
        store.log_meal(pid,(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(),items,ctx,pred)
    ident=store.log_meal(pid,datetime.now(timezone.utc).isoformat(),items,ctx,pred)
    with pytest.raises(ValueError,match="has not arrived"):
        store.record_outcome(pid,ident,150,confirmed=True)
    ident=store.log_meal(pid,(datetime.now(timezone.utc)-timedelta(days=1)).isoformat(),items,ctx,pred)
    reason=store.record_outcome(pid,ident,150,minute120=200,confirmed=True)
    assert reason
    assert not eligible(store.meals(pid),pred["version"])


def test_custom_food_scope_and_validation(store):
    a,b=store.create_profile("A"),store.create_profile("B")
    nutrition={"energy_kcal":150,"carb_g":20,"protein_g":5,"fat_g":5,"fibre_g":3,"sugar_g":1}
    store.add_food(a,"My dosa",nutrition,100)
    assert len(store.custom_foods(a))==1
    assert not store.custom_foods(b)
    with pytest.raises(ValueError,match="already"):
        store.add_food(a,"My dosa",nutrition,100)
    with pytest.raises(ValueError):
        store.add_food(a,"Bad food",{**nutrition,"carb_g":float("nan")},100)
    with pytest.raises(ValueError):
        store.add_food(a,"Bad food",{**nutrition,"sugar_g":70},100)


def test_incremental_measurements_preserve_previous_values(store,meal):
    pid=store.create_profile("Incremental")
    context,items,pred=meal
    ident=store.log_meal(pid,(datetime.now(timezone.utc)-timedelta(days=1)).isoformat(),items,context,pred)
    for minute,value in [(15,120),(30,140),(45,155),(60,150),(120,130)]:
        store.record_readings(pid,ident,{minute:value},confirmed=True)
        saved=store.meals(pid)[0]
        assert saved[f"observed{minute}"]==value
        if minute<120:
            assert not eligible(store.meals(pid),pred["version"])
    initial=store.meals(pid)[0]
    assert initial["observed15"]==120
    assert len(eligible(store.meals(pid),pred["version"]))==1
    with pytest.raises(ValueError,match="already has"):
        store.record_readings(pid,ident,{15:180},confirmed=True)
    # Schema initialization is idempotent and preserves every saved timepoint.
    assert Store(store.path).meals(pid)==[initial]


def test_reading_batch_is_atomic_and_scoped(store,meal):
    a,b=store.create_profile("A"),store.create_profile("B")
    ctx,items,pred=meal
    ident=store.log_meal(a,(datetime.now(timezone.utc)-timedelta(minutes=20)).isoformat(),items,ctx,pred)
    with pytest.raises(ValueError,match="not found"):
        store.record_readings(b,ident,{15:125},confirmed=True)
    with pytest.raises(ValueError,match="has not arrived"):
        store.record_readings(a,ident,{15:125,120:140},confirmed=True)
    assert store.meals(a)[0]["observed15"] is None
    store.record_readings(a,ident,{15:125},confirmed=True)
    with pytest.raises(ValueError,match="increase"):
        store.record_readings(a,ident,{30:140},{30:10},confirmed=True)
    with pytest.raises(ValueError):
        store.record_readings(a,ident,{30:float("nan")},confirmed=True)
    with pytest.raises(ValueError):
        store.record_readings(a,ident,{90:140},confirmed=True)
    assert store.meals(a)[0]["observed30"] is None


def test_existing_two_point_database_migrates_without_changing_readings(store,meal):
    pid=store.create_profile("Existing profile")
    ctx,items,pred=meal
    ident=store.log_meal(pid,(datetime.now(timezone.utc)-timedelta(days=1)).isoformat(),items,ctx,pred)
    store.record_outcome(pid,ident,130,150,confirmed=True)
    before=store.meals(pid)[0]
    # Recreate the previous schema on this isolated test database.
    with store.connection() as con:
        for name in ["observed15","observed30","observed45","minute15","minute30","minute45",
                     "measured_at15","measured_at30","measured_at45","measured_at60","measured_at120"]:
            con.execute(f"ALTER TABLE meals DROP COLUMN {name}")
    upgraded=Store(store.path)
    after=upgraded.meals(pid)[0]
    for name in ["observed60","observed120","minute60","minute120","measured_at","items","context","model_version"]:
        assert after[name]==before[name]
    upgraded.record_readings(pid,ident,{15:120,30:135,45:145},confirmed=True)
    assert upgraded.meals(pid)[0]["observed120"]==130


def measured_rows(meal,count=24,shift=25):
    ctx,items,pred=meal
    result=[]
    for i in range(count):
        eaten=datetime(2026,8,1,tzinfo=timezone.utc)+timedelta(days=i)
        result.append({"id":f"meal_{i}","eaten_at":eaten.isoformat(),"measured_at":(eaten+timedelta(hours=2)).isoformat(),
            "items":copy.deepcopy(items),"context":dict(ctx),"base60":pred["base"][0],"base120":pred["base"][1],
            "observed60":pred["base"][0]+shift,"observed120":pred["base"][1]+shift,
            "model_version":pred["version"],"confounded":0,"excluded_reason":"","provenance":"measured"})
    return result


def test_measured_only_and_activation(meal):
    rows=measured_rows(meal)
    state=fit_personal(rows,meal[2]["version"])
    assert state["targets"][120]["active"]
    assert apply_personal(state,meal[0],meal[1])[1]>15
    contaminated=copy.deepcopy(rows)
    for row in contaminated:
        row["provenance"]="synthetic"
    state=fit_personal(contaminated,meal[2]["version"])
    assert state["eligible_meals"]==0
    assert np.all(apply_personal(state,meal[0],meal[1])==0)


def test_backfilled_labels_cannot_leak_to_earlier_validation(meal):
    rows=measured_rows(meal)
    for row in rows:
        row["measured_at"]="2026-10-18T10:00:00+00:00"
    state=fit_personal(rows,meal[2]["version"])
    assert not state["targets"][120]["active"]


def test_late_one_hour_entry_does_not_borrow_two_hour_recording_time(meal):
    rows=measured_rows(meal)
    for row in rows:
        row["measured_at120"]=row["measured_at"]
        row["measured_at60"]="2026-10-18T10:00:00+00:00"
    state=fit_personal(rows,meal[2]["version"])
    assert not state["targets"][60]["active"]
    assert state["targets"][120]["active"]


def test_no_improvement_retains_base(meal):
    state=fit_personal(measured_rows(meal,shift=0),meal[2]["version"])
    assert not state["targets"][120]["active"]


def test_evidence_respects_portions_and_low_outcomes(meal):
    rows=measured_rows(meal,count=4)
    for r in rows:
        r["observed120"]=130
    evidence=measured_evidence(rows,meal[1],meal[0],70,180)
    assert evidence["n"]==4
    status,_=classify(125,90,160,evidence,70,180)
    assert status=="Likely suitable"
    rows[0]["observed120"]=60
    evidence=measured_evidence(rows,meal[1],meal[0],70,180)
    assert classify(125,90,160,evidence,70,180)[0]=="Review with care team"
    larger=copy.deepcopy(meal[1])
    larger[0]["grams"]*=2
    assert measured_evidence(rows,larger,meal[0],70,180)["n"]==0

"""Local SQLite storage with immutable meal snapshots and profile-scoped reads/writes."""
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

from twin.config import DATA, FEATURES, NUTRIENTS

READING_MINUTES = (15, 30, 45, 60, 120)


def utcnow():
    return datetime.now(timezone.utc)


def parse_time(value):
    result = datetime.fromisoformat(str(value))
    if result.tzinfo is None:
        raise ValueError("A timestamp must include its time zone.")
    return result.astimezone(timezone.utc)


def finite(value, low, high, label):
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be between {low} and {high}.")
    return value


class Store:
    def __init__(self, path=None):
        self.path = Path(path or DATA / "local/twin_v2.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, diabetes_type TEXT NOT NULL,
                    target_low REAL NOT NULL, target_high REAL NOT NULL,
                    restrictions TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS custom_foods (
                    food_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id),
                    name TEXT NOT NULL, record_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(profile_id, name)
                );
                CREATE TABLE IF NOT EXISTS meals (
                    id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id),
                    eaten_at TEXT NOT NULL, created_at TEXT NOT NULL, items_json TEXT NOT NULL,
                    context_json TEXT NOT NULL, base60 REAL NOT NULL, base120 REAL NOT NULL,
                    pred60 REAL NOT NULL, pred120 REAL NOT NULL, model_version TEXT NOT NULL,
                    personal_version TEXT NOT NULL DEFAULT '', low120 REAL, high120 REAL,
                    observed60 REAL, observed120 REAL, minute60 REAL, minute120 REAL,
                    measured_at TEXT, confounded INTEGER NOT NULL DEFAULT 0,
                    excluded_reason TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'measured',
                    UNIQUE(profile_id, eaten_at)
                );
                CREATE TABLE IF NOT EXISTS learning_runs (
                    id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id),
                    created_at TEXT NOT NULL, details_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS profile_meals ON meals(profile_id, eaten_at);
            """)
            # Additive migration applies only to this app's database, never the preserved legacy database.
            columns={r["name"] for r in con.execute("PRAGMA table_info(meals)")}
            additions=[("personal_version","TEXT NOT NULL DEFAULT ''"),("low120","REAL"),("high120","REAL")]
            additions += [(f"observed{m}","REAL") for m in (15,30,45)]
            additions += [(f"minute{m}","REAL") for m in (15,30,45)]
            additions += [(f"measured_at{m}","TEXT") for m in READING_MINUTES]
            for name,definition in additions:
                if name not in columns:
                    con.execute(f"ALTER TABLE meals ADD COLUMN {name} {definition}")

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.path, timeout=20)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def profiles(self):
        with self.connection() as con:
            return [dict(r) for r in con.execute("SELECT * FROM profiles ORDER BY created_at")]

    def get_profile(self, profile_id):
        with self.connection() as con:
            row = con.execute("SELECT * FROM profiles WHERE id=?", (profile_id,)).fetchone()
        if row is None:
            raise ValueError("Profile not found. Select a local profile.")
        return dict(row)

    def create_profile(self, name, diabetes_type="Type 2", target_low=70, target_high=180, restrictions=""):
        name = str(name).strip()
        if not 1 <= len(name) <= 60:
            raise ValueError("Use a profile name of 1 to 60 characters.")
        low, high = finite(target_low,70,120,"Lower target"), finite(target_high,120,220,"Upper target")
        if low >= high:
            raise ValueError("The upper target must exceed the lower target.")
        if diabetes_type not in ["Type 1", "Type 2", "Prediabetes", "Other / unsure"]:
            raise ValueError("Unknown profile type.")
        if len(str(restrictions)) > 1500:
            raise ValueError("Keep restrictions under 1,500 characters.")
        profile_id = uuid.uuid4().hex
        with self.connection() as con:
            con.execute("INSERT INTO profiles VALUES (?,?,?,?,?,?,?)", (profile_id,name,diabetes_type,low,high,str(restrictions),utcnow().isoformat()))
        return profile_id

    def custom_foods(self, profile_id):
        with self.connection() as con:
            rows = con.execute("SELECT record_json FROM custom_foods WHERE profile_id=?",(profile_id,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def update_targets(self, profile_id, low, high, restrictions):
        self.get_profile(profile_id)
        low, high = finite(low,70,120,"Lower target"), finite(high,120,220,"Upper target")
        if low >= high or len(restrictions)>1500:
            raise ValueError("Check target order and keep restrictions under 1,500 characters.")
        with self.connection() as con:
            con.execute("UPDATE profiles SET target_low=?,target_high=?,restrictions=? WHERE id=?",(low,high,restrictions,profile_id))

    def add_food(self, profile_id, name, nutrients, serving_g, notes=""):
        self.get_profile(profile_id)
        name = str(name).strip()
        if not 2 <= len(name) <= 100:
            raise ValueError("Food name must have 2 to 100 characters.")
        serving_g = finite(serving_g,5,750,"Serving weight")
        if len(notes)>1000:
            raise ValueError("Use a shorter source note.")
        clean = {k:finite(nutrients[k],0,950 if k=="energy_kcal" else 100,k) for k in NUTRIENTS}
        if clean["sugar_g"] > clean["carb_g"] or sum(clean[k] for k in ["carb_g","protein_g","fat_g"]) > 105:
            raise ValueError("Check per-100 g values: sugar must not exceed carbs, and macros must fit within 100 g.")
        food_id = "custom_"+uuid.uuid4().hex
        record = {"food_id":food_id,"name":name,"family_id":food_id,"source_code":"USER",
                  "source":"Your food","primary_source":"user_entered","source_url":"",
                  "evidence":"user-entered nutrition; unverified","category":"Your foods","is_recipe":True,
            "meal_eligible":True,"serving_g":serving_g,"serving_unit":"weighed serving","serving_basis":"user-entered mass",
                  "notes":notes,**clean}
        try:
            with self.connection() as con:
                con.execute("INSERT INTO custom_foods VALUES (?,?,?,?,?)",(food_id,profile_id,name,json.dumps(record),utcnow().isoformat()))
        except sqlite3.IntegrityError as exc:
            raise ValueError("This profile already has a food with that name.") from exc
        return record

    def log_meal(self, profile_id, eaten_at, items, context, prediction):
        self.get_profile(profile_id)
        timestamp = parse_time(eaten_at)
        if timestamp > utcnow()+timedelta(minutes=5):
            raise ValueError("Only log a meal that has already started.")
        if not items or len(items)>12:
            raise ValueError("Log between 1 and 12 food items.")
        for k in FEATURES:
            finite(context[k],0,600 if k=="baseline" else 2000,k)
        if prediction.get("blocked"):
            raise ValueError("This meal cannot be logged from a blocked estimate.")
        for item in items:
            finite(item["grams"],5,1000,"Food weight")
        meal_id = uuid.uuid4().hex
        try:
            with self.connection() as con:
                con.execute("""INSERT INTO meals
                    (id,profile_id,eaten_at,created_at,items_json,context_json,base60,base120,pred60,pred120,model_version,personal_version,low120,high120)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (meal_id,profile_id,timestamp.isoformat(),utcnow().isoformat(),json.dumps(items),json.dumps(context),
                     *[float(v) for v in prediction["base"]], *[float(v) for v in prediction["mean"]],prediction["version"],
                     prediction["personal_version"],float(prediction["lower"][1]),float(prediction["upper"][1])))
        except sqlite3.IntegrityError as exc:
            raise ValueError("A meal is already logged at this time. Use its existing record.") from exc
        return meal_id

    def meals(self, profile_id):
        with self.connection() as con:
            rows = con.execute("SELECT * FROM meals WHERE profile_id=? ORDER BY eaten_at,created_at",(profile_id,)).fetchall()
        result=[]
        for row in rows:
            value=dict(row)
            value["items"]=json.loads(value.pop("items_json"))
            value["context"]=json.loads(value.pop("context_json"))
            result.append(value)
        return result

    def record_outcome(self, profile_id, meal_id, g120, g60=None, minute120=120, minute60=60, confirmed=False, confounded=False):
        """Compatibility entry point for the original one/two-hour form."""
        readings={120:g120}
        if g60 is not None:
            readings[60]=g60
        return self.record_readings(profile_id,meal_id,readings,{60:minute60,120:minute120},confirmed,confounded)

    def record_readings(self, profile_id, meal_id, readings, actual_minutes=None, confirmed=False, confounded=False):
        """Append any available timepoints atomically; saved measurements cannot be overwritten."""
        if confirmed is not True:
            raise ValueError("Confirm these are measured glucose readings, not predictions or sample values.")
        if not readings or any(m not in READING_MINUTES for m in readings):
            raise ValueError("Enter at least one reading at 15, 30, 45, 60 or 120 minutes.")
        clean={m:finite(value,20,600,f"{m}-minute glucose") for m,value in readings.items()}
        actual_minutes=actual_minutes or {}
        timing={m:finite(actual_minutes.get(m,m),1,300,"Actual minutes after meal") for m in clean}
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row=con.execute("SELECT * FROM meals WHERE id=? AND profile_id=?",(meal_id,profile_id)).fetchone()
            if row is None:
                raise ValueError("Meal not found for this profile.")
            combined=dict(row)
            stamp=utcnow()
            for minute,value in clean.items():
                if row[f"observed{minute}"] is not None:
                    raise ValueError(f"This meal already has a {minute}-minute reading. Saved readings cannot be overwritten.")
                if parse_time(row["eaten_at"])+timedelta(minutes=timing[minute])>stamp+timedelta(minutes=5):
                    raise ValueError("The measurement time has not arrived. Save the actual reading after the meal.")
                combined[f"observed{minute}"]=value
                combined[f"minute{minute}"]=timing[minute]
                combined[f"measured_at{minute}"]=stamp.isoformat()
            sequence=[combined[f"minute{m}"] for m in READING_MINUTES if combined[f"observed{m}"] is not None]
            if any(a>=b for a,b in zip(sequence,sequence[1:])):
                raise ValueError("Actual measurement times must increase from the earlier to the later reading.")
            reasons=[]
            confounded=bool(confounded or row["confounded"])
            if confounded:
                reasons.append("illness, treatment change or extra food reported")
            if combined["observed120"] is not None and not 105 <= combined["minute120"] <= 135:
                reasons.append("later reading outside 105-135 minute window")
            if combined["observed60"] is not None and not 50 <= combined["minute60"] <= 70:
                reasons.append("earlier reading outside 50-70 minute window")
            baseline=json.loads(row["context_json"])["baseline"]
            if baseline < 80 or baseline > 240:
                reasons.append("premeal glucose outside model range")
            assignments=[]
            values=[]
            for minute in clean:
                # Column names are restricted to the constant READING_MINUTES allowlist above.
                assignments.extend([f"observed{minute}=?",f"minute{minute}=?",f"measured_at{minute}=?"])
                values.extend([combined[f"observed{minute}"],combined[f"minute{minute}"],stamp.isoformat()])
            availability=stamp.isoformat() if any(m in clean for m in (60,120)) else row["measured_at"]
            assignments.extend(["measured_at=?","confounded=?","excluded_reason=?"])
            values.extend([availability,int(confounded),"; ".join(reasons),meal_id,profile_id])
            con.execute("UPDATE meals SET "+",".join(assignments)+" WHERE id=? AND profile_id=?",values)
        return reasons

    def audit_learning(self, profile_id, details):
        self.get_profile(profile_id)
        with self.connection() as con:
            con.execute("INSERT INTO learning_runs VALUES (?,?,?,?)",(uuid.uuid4().hex,profile_id,utcnow().isoformat(),json.dumps(details)))

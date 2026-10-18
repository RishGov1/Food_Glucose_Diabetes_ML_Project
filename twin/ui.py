import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from twin.config import DATA, FEATURES, MODELS, NUTRIENTS, REPORTS, ROOT
from twin.learning import fit_personal, learning_summary
from twin.service import glucose_to_mg, load_catalog, meal_totals, predict_meal, safety_message
from twin.storage import Store, parse_time, READING_MINUTES

LOCAL_TZ = timezone(timedelta(hours=5,minutes=30))
DEMO = {"id":"demo","name":"Explore an example","diabetes_type":"Type 2","target_low":70.,"target_high":180.,"restrictions":""}


@st.cache_resource
def artifacts(version):
    return joblib.load(MODELS/"meal_prior.joblib")


def safe_csv(frame):
    """Prevent spreadsheet formula execution in local exports of user-entered text."""
    frame=frame.copy()
    for col in frame.columns:
        frame[col]=frame[col].map(lambda x: "'"+x if isinstance(x,str) and x.lstrip().startswith(("=","+","-","@")) else x)
    return frame.to_csv(index=False).encode("utf-8-sig")


def toast_result(message):
    st.session_state["flash"]=message
    st.rerun()


def sidebar(store):
    with st.sidebar:
        st.markdown("### Glucose Twin")
        st.caption("Your meals. Your measured response.")
        profiles=store.profiles()
        options=["demo"]+[p["id"] for p in profiles]
        lookup={p["id"]:p for p in profiles}
        lookup["demo"]=DEMO
        if st.session_state.get("next_profile"):
            st.session_state["profile_select"]=st.session_state.pop("next_profile")
        selected=st.selectbox("Local profile",options,format_func=lambda k:lookup[k]["name"],key="profile_select")
        profile=lookup[selected]
        if selected=="demo":
            st.caption("Example inputs only. Nothing here trains a personal model.")
        with st.expander("Create a personal profile"):
            with st.form("create_profile"):
                name=st.text_input("Profile nickname",max_chars=60)
                kind=st.selectbox("Diabetes type",["Type 2","Type 1","Prediabetes","Other / unsure"])
                st.caption("For adult research use. Pregnancy, children and treatment decisions need a clinical care plan.")
                if st.form_submit_button("Create profile",width="stretch"):
                    try:
                        st.session_state["next_profile"]=store.create_profile(name,kind)
                        toast_result("Your local profile is ready.")
                    except ValueError as exc:
                        st.error(str(exc))
        if st.session_state.get("next_page"):
            st.session_state["page"]=st.session_state.pop("next_page")
        page=st.radio("Workspace",["Meal check","Log a meal","Meal journal","My food lists","Add a food","Model & data"],label_visibility="collapsed",key="page")
        st.divider()
        st.markdown("**Meal context**")
        unit=st.selectbox("Glucose units",["mg/dL","mmol/L"],key=f"units_{selected}")
        baseline_value=st.number_input("Before-meal glucose",min_value=20. if unit=="mg/dL" else 1.11,
            max_value=600. if unit=="mg/dL" else 33.29,value=110. if unit=="mg/dL" else 6.1,
            step=1. if unit=="mg/dL" else .1,key=f"baseline_{selected}_{unit}")
        baseline=glucose_to_mg(baseline_value,unit)
        with st.expander("Sleep and recent activity"):
            activity=st.slider("Activity in the previous 2 hours (minutes)",0,60,0,step=5,key=f"activity_{selected}")
            sleep=st.slider("Last night's sleep (hours)",4.,10.,7.,step=.5,key=f"sleep_{selected}")
        st.caption(f"Two-hour target: {profile['target_low']:g} to under {profile['target_high']:g} mg/dL")
        if selected!="demo":
            with st.expander("My targets and care notes"):
                with st.form("targets_"+selected):
                    st.caption("Use targets agreed with your clinician. Notes are reminders, not automatically checked for allergens or interactions.")
                    low=st.number_input("Lower target (mg/dL)",70.,120.,float(profile["target_low"]))
                    high=st.number_input("Upper target (mg/dL)",120.,220.,float(profile["target_high"]))
                    notes=st.text_area("Allergies / care-team instructions",value=profile["restrictions"],max_chars=1500)
                    if st.form_submit_button("Save targets"):
                        try:
                            store.update_targets(selected,low,high,notes)
                            toast_result("Targets and care notes updated.")
                        except ValueError as exc:
                            st.error(str(exc))
        st.divider()
        st.caption("Stored on this computer. No account or cloud sync. Use a trusted personal device.")
        st.caption("Research prototype · Not for insulin, medication or emergency decisions.")
    return profile,page,{"baseline":baseline,"activity_min":activity,"sleep_hours":sleep}


def learning_state(rows,artifact,profile):
    token=(profile["id"],artifact["report"]["version"],hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
    if st.session_state.get("learning_token")!=token:
        st.session_state["personal"]=fit_personal(rows,artifact["report"]["version"])
        st.session_state["learning_token"]=token
    return st.session_state["personal"]


def context_caption(context):
    return f"Premeal {context['baseline']:.0f} mg/dL · recent activity {context['activity_min']:g} min · sleep {context['sleep_hours']:g} h"


def estimate_chart(pred,context,profile):
    fig=go.Figure()
    fig.add_hrect(y0=profile["target_low"],y1=profile["target_high"],fillcolor="#CDE9DD",opacity=.35,line_width=0)
    fig.add_trace(go.Scatter(x=[0],y=[context["baseline"]],mode="markers",name="Entered premeal reading",marker={"color":"#183A36","size":11}))
    fig.add_trace(go.Scatter(x=[60,120],y=pred["mean"],mode="markers",name="Illustrative estimates",
        marker={"color":"#147D73","size":12},error_y={"type":"data","symmetric":False,
        "array":np.array(pred["upper"])-pred["mean"],"arrayminus":np.array(pred["mean"])-pred["lower"],"thickness":2,"width":7}))
    fig.update_layout(height=300,margin={"t":15,"b":20,"l":10,"r":10},paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#183A36"},xaxis_title="Minutes after the meal starts",yaxis_title="Glucose (mg/dL)",legend={"orientation":"h","y":1.18})
    fig.update_xaxes(tickvals=[0,60,120],gridcolor="#E4ECE7")
    fig.update_yaxes(gridcolor="#E4ECE7")
    st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})


def meal_page(store,profile,context,catalog,artifact,personal,rows,log_only=False):
    st.title("Log the foods you ate" if log_only else "A clearer picture of your next meal")
    st.write("Select all foods in the same meal, enter their portions and your premeal reading, then save the meal. Add follow-up readings in Meal journal." if log_only else "Explore a portion, see its nutrition, then let measured outcomes build your personal evidence.")
    if profile["id"]=="demo":
        st.info("You’re exploring example inputs. Create a personal profile to save meals and learn from your own readings.")
    elif not log_only:
        if st.button("I already ate — log my meal"):
            st.session_state["next_page"]="Log a meal"
            st.rerun()
    recipes=catalog[catalog.meal_eligible].copy()
    lookup=recipes.set_index("food_id")
    default=recipes[recipes.name.str.contains("idli",case=False)].head(1).food_id.tolist()
    selected=st.multiselect("What is in the meal?",recipes.food_id.tolist(),default=default,
        format_func=lambda k:lookup.loc[k,"name"],max_selections=12,key="selected_foods_"+profile["id"])
    portions=[]
    if selected:
        cols=st.columns(min(3,len(selected)))
        for i,food_id in enumerate(selected):
            food=lookup.loc[food_id]
            with cols[i%len(cols)]:
                grams=st.number_input(f"{food['name']} · grams",5.,1000.,float(np.clip(food.serving_g,5,1000)),step=5.,key=f"weight_{profile['id']}_{food_id}")
                st.caption(f"Reference: {food.serving_g:g} g · {food.serving_unit}")
                portions.append((food_id,grams))
    else:
        st.info("Choose a cooked food above, or add your own food with nutrition per 100 g.")
        return
    totals,items=meal_totals(catalog,portions)
    current={**context,**{k:totals[k] for k in FEATURES if k in totals}}
    a,b,c,d=st.columns(4)
    a.metric("Energy",f"{totals['energy_kcal']:.0f} kcal")
    b.metric("Carbohydrate",f"{totals['carb_g']:.1f} g")
    c.metric("Protein",f"{totals['protein_g']:.1f} g")
    d.metric("Fibre",f"{totals['fibre_g']:.1f} g")
    with st.expander("Food details and sources"):
        st.dataframe(pd.DataFrame(items)[["name","grams",*NUTRIENTS,"evidence","source"]],hide_index=True,width="stretch")
        st.caption("Values depend on recipe and preparation. Free sugar is shown when provided; it is not total sugar. Weighed portions are more reliable than bowl or cup labels.")
    signature=hashlib.sha256(json.dumps([profile,personal["version"],current,items],sort_keys=True).encode()).hexdigest()
    if log_only:
        if profile["id"]!="demo":
            pred=predict_meal(artifact,personal,current,items,rows,profile)
            if pred["blocked"]:
                st.warning(pred["reason"])
            else:
                save_meal_form(store,profile,items,current,pred)
        return
    if st.button("Check this meal",type="primary",width="stretch"):
        st.session_state["estimate"]={"signature":signature,"result":predict_meal(artifact,personal,current,items,rows,profile)}
    saved=st.session_state.get("estimate")
    if not saved or saved["signature"]!=signature:
        st.caption("Check the meal after changing foods, portions, readings or targets.")
        return
    pred=saved["result"]
    if pred["blocked"]:
        st.warning(pred["reason"])
        return
    st.divider()
    left,right=st.columns([1,1.7])
    with left:
        st.subheader(pred["status"])
        st.write(pred["reason"])
        st.metric("Illustrative two-hour glucose",f"{pred['mean'][1]:.0f} mg/dL")
        st.write(f"Illustrative range: **{pred['lower'][1]:.0f}–{pred['upper'][1]:.0f} mg/dL**")
        st.caption("This model starts from simulated responses. The range is not a clinically validated confidence interval. Verify with your own measurements.")
        st.caption(f"Comparable measured meals: {pred['evidence']['n']} · Personal adjustment: {pred['adjustment'][1]:+.1f} mg/dL")
    with right:
        estimate_chart(pred,current,profile)
    st.caption(context_caption(current))
    if pred["evidence"]["n"]:
        st.write(f"Your comparable outcomes: {pred['evidence']['in_target']} within target, {pred['evidence']['above']} above and {pred['evidence']['below']} below.")
    with st.expander("Compare portions"):
        comparison=[]
        for factor in [.5,.75,1,1.25]:
            if not all(5<=g*factor<=1000 for _,g in portions):
                continue
            t,it=meal_totals(catalog,[(f,g*factor) for f,g in portions])
            ct={**context,**{k:t[k] for k in FEATURES if k in t}}
            alt=predict_meal(artifact,personal,ct,it,rows,profile)
            comparison.append({"Portion":f"{factor*100:.0f}%","Carbs (g)":round(t["carb_g"],1),
                               "Illustrative 2h (mg/dL)":None if alt["blocked"] else round(alt["mean"][1]),
                               "Assessment":alt["reason"] if alt["blocked"] else alt["status"]})
        st.dataframe(pd.DataFrame(comparison),hide_index=True,width="stretch")
        st.caption("A portion comparison is a model scenario, not a prescription. Keep to your care plan and nutritional needs.")
    if profile["id"]=="demo":
        return
    save_meal_form(store,profile,items,current,pred)


def save_meal_form(store,profile,items,current,pred):
    st.subheader("Save a meal you ate")
    with st.form("log_meal_"+profile["id"]):
        now=datetime.now(LOCAL_TZ)
        col1,col2=st.columns(2)
        day=col1.date_input("Meal date",now.date(),max_value=now.date())
        when=col2.time_input("Meal start time (India)",now.time().replace(second=0,microsecond=0))
        confirm=st.checkbox("I ate this portion and entered my measured premeal glucose.")
        if st.form_submit_button("Save meal to journal",type="primary"):
            if not confirm:
                st.error("Confirm the meal and premeal measurement before saving.")
            else:
                try:
                    store.log_meal(profile["id"],datetime.combine(day,when,tzinfo=LOCAL_TZ).isoformat(),items,current,pred)
                    st.session_state["next_page"]="Meal journal"
                    toast_result("Meal saved. Add readings at 15, 30, 45, 60 minutes (1 hour) and 120 minutes (2 hours) as they become available.")
                except ValueError as exc:
                    st.error(str(exc))


def foods_page(profile,context,catalog,artifact,personal,rows):
    st.title("Your food lists")
    st.write("A personal shortlist for the current portion and glucose reading. Foods are never permanently banned by this app.")
    st.caption("‘Likely suitable’ is your provisional allowed list. ‘Review portion’ and ‘Limit / discuss’ form your review list. With no repeated personal measurements, foods remain unconfirmed.")
    st.caption(context_caption(context))
    search=st.text_input("Search Indian foods",placeholder="Try dosa, rajma, rice, paneer…")
    col1,col2,col3=st.columns([1.1,1.1,1])
    category=col1.selectbox("Food category",["All categories"]+sorted(catalog.category.unique().tolist()))
    source_kind=col2.selectbox("Show",["Recipes, fruit and my foods","All foods, including ingredients"])
    multiplier=col3.selectbox("Reference portion",[.5,.75,1.,1.5,2.],index=2,format_func=lambda x:f"{x:g} × listed serving")
    view=catalog.copy()
    if source_kind.startswith("Recipes"):
        view=view[view.meal_eligible]
    if search:
        view=view[view.name.str.contains(search,case=False,regex=False)]
    if category!="All categories":
        view=view[view.category==category]
    if view.empty:
        st.info("No foods match. Try another spelling or add a food with its nutrition label.")
        return
    records=[]
    with st.spinner("Checking food portions…"):
        for _,food in view.iterrows():
            grams=float(food.serving_g)*multiplier
            if not 5<=grams<=1000:
                continue
            totals,items=meal_totals(catalog,[(food.food_id,grams)])
            ctx={**context,**{k:totals[k] for k in FEATURES if k in totals}}
            p=predict_meal(artifact,personal,ctx,items,rows,profile)
            records.append({"Food":food["name"],"Assessment":"Not assessed" if p["blocked"] else p["status"],"Portion (g)":round(grams,1),
                "Carbs (g)":round(totals["carb_g"],1),"Fibre (g)":round(totals["fibre_g"],1),"Protein (g)":round(totals["protein_g"],1),
                "Fat (g)":round(totals["fat_g"],1),"Free sugar (g)":round(totals["sugar_g"],1),"Energy (kcal)":round(totals["energy_kcal"]),
                "Illustrative 2h":None if p["blocked"] else round(p["mean"][1]),
                "Comparable meals":0 if p["blocked"] else p["evidence"]["n"],"Reason":p["reason"],"Evidence":food.evidence,"Source":food.source,"Source link":food.source_url})
    frame=pd.DataFrame(records)
    if frame.empty:
        st.info("These reference portions are outside the supported weight range. Choose a smaller multiplier.")
        return
    a,b,c=st.columns(3)
    a.metric("Likely suitable",int((frame.Assessment=="Likely suitable").sum()))
    b.metric("Review / limit",int(frame.Assessment.isin(["Review portion","Limit / discuss","Review with care team"]).sum()))
    c.metric("Unconfirmed / unassessed",int(frame.Assessment.isin(["Promising / unconfirmed","Uncertain","Not assessed"]).sum()))
    status=st.multiselect("Filter assessments",frame.Assessment.unique().tolist())
    if status:
        frame=frame[frame.Assessment.isin(status)]
    st.dataframe(frame,hide_index=True,width="stretch",height=470,column_config={"Source link":st.column_config.LinkColumn("Source")})
    st.download_button("Download this food list",safe_csv(frame),file_name="my_food_review.csv",mime="text/csv")
    st.caption("Ingredients are shown in their source state and are not automatically ranked as cooked meals. Nutrition estimates do not check allergies, kidney disease, interactions or your full daily diet.")


def history_table(rows):
    return pd.DataFrame([{"Meal ID":r["id"],"Meal time (India)":parse_time(r["eaten_at"]).astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
        "Foods":", ".join(f"{i['name']} ({i['grams']:g} g)" for i in r["items"]),"Premeal":r["context"]["baseline"],
        "Predicted 60":r["pred60"],"Predicted 120":r["pred120"],
        **{f"Measured {m}":r.get(f"observed{m}") for m in READING_MINUTES},
        **{f"Actual minutes for {m}":r.get(f"minute{m}") for m in READING_MINUTES},
        "Learning exclusion":r["excluded_reason"],"Base model":r["model_version"]} for r in rows])


def journal_page(store,profile,rows,artifact,personal):
    st.title("Your meal journal")
    st.write("Select a meal and add readings at 15, 30, 45, 60 minutes (1 hour) and 120 minutes (2 hours). Save each as you take it, or enter several together.")
    if profile["id"]=="demo":
        st.info("Create a personal profile to log real meals. Example data never enters personal training.")
        return
    measured=[r for r in rows if r["observed120"] is not None]
    active=personal["targets"][120]["active"]
    a,b,c=st.columns(3)
    a.metric("Logged meals",len(rows))
    b.metric("Measured outcomes",len(measured))
    c.metric("Personal learning","Active" if active else "Collecting / checking")
    st.caption(personal["targets"][120]["reason"])
    if st.button("Log another meal"):
        st.session_state["next_page"]="Log a meal"
        st.rerun()
    pending=[r for r in rows if any(r.get(f"observed{m}") is None for m in READING_MINUTES)]
    if pending:
        st.subheader("Add glucose readings for a meal")
        options={r["id"]:r for r in pending}
        meal_id=st.selectbox("Meal awaiting a reading",list(options),format_func=lambda k:
            parse_time(options[k]["eaten_at"]).astimezone(LOCAL_TZ).strftime("%d %b %H:%M")+" · "+", ".join(i["name"] for i in options[k]["items"]))
        chosen=options[meal_id]
        st.write("Foods: "+", ".join(f"{i['name']} ({i['grams']:g} g)" for i in chosen["items"]))
        st.caption(f"Premeal glucose: {chosen['context']['baseline']:.1f} mg/dL. All times are measured from the start of this meal.")
        unit=st.selectbox("Outcome units",["mg/dL","mmol/L"],key="reading_units_"+meal_id)
        with st.form("outcome_"+meal_id):
            values={}
            timing={}
            labels={15:"15-minute glucose",30:"30-minute glucose",45:"45-minute glucose",60:"60-minute glucose (1 hour)",120:"120-minute glucose (2 hours)"}
            columns=st.columns(3)
            for i,m in enumerate(READING_MINUTES):
                saved=chosen.get(f"observed{m}")
                shown=saved/18.0182 if saved is not None and unit=="mmol/L" else saved
                values[m]=columns[i%3].number_input(labels[m],value=shown,min_value=0.,max_value=600.,
                    disabled=saved is not None,placeholder="Leave blank until measured",key=f"reading_{meal_id}_{m}_{unit}_{saved}")
                if saved is not None:
                    columns[i%3].caption(f"Saved · taken at {chosen[f'minute{m}']:g} minutes")
            with st.expander("Actual measurement times"):
                st.caption("If a measurement was taken a little earlier or later, enter its actual elapsed minutes here.")
                for m in READING_MINUTES:
                    timing[m]=st.number_input(f"Actual minutes for the {m}-minute reading",1.,300.,float(chosen.get(f"minute{m}") or m),
                        disabled=chosen.get(f"observed{m}") is not None,key=f"timing_{meal_id}_{m}_{chosen.get(f'minute{m}')}")
            changed=st.checkbox("Illness, a treatment change, or extra food affected this meal",value=bool(chosen["confounded"]),disabled=bool(chosen["confounded"]))
            confirm=st.checkbox("These are actual meter / CGM readings, not predicted or demonstration values.")
            st.caption("Save whichever readings you have; no need to wait two hours. The 15/30/45-minute readings are stored and plotted. The current learning model uses eligible 60/120-minute outcomes after a two-hour outcome is available.")
            if st.form_submit_button("Save readings and update twin",type="primary"):
                try:
                    new={m:glucose_to_mg(v,unit) for m,v in values.items() if v is not None and chosen.get(f"observed{m}") is None}
                    exclusions=store.record_readings(profile["id"],meal_id,new,timing,confirm,changed)
                    refreshed=fit_personal(store.meals(profile["id"]),artifact["report"]["version"])
                    store.audit_learning(profile["id"],learning_summary(refreshed))
                    st.session_state["outcome_alert"]=next((safety_message(v) for v in new.values() if safety_message(v)),"")
                    toast_result("Readings saved. "+("Kept in history; excluded from learning: "+"; ".join(exclusions) if exclusions else "You can return to add the remaining readings."))
                except ValueError as exc:
                    st.error(str(exc))
    elif not rows:
        st.info("Choose Log a meal in the sidebar, select what you ate and save it. Then add your readings here.")
    else:
        st.success("All five follow-up readings are saved for every logged meal.")
    if rows:
        st.subheader("Measured response for a meal")
        lookup={r["id"]:r for r in rows}
        selected=st.selectbox("Meal to plot",list(lookup),format_func=lambda k:
            parse_time(lookup[k]["eaten_at"]).astimezone(LOCAL_TZ).strftime("%d %b %H:%M")+" · "+", ".join(i["name"] for i in lookup[k]["items"]))
        r=lookup[selected]
        points=[{"Minutes after meal":0.,"Glucose (mg/dL)":r["context"]["baseline"]}]
        points.extend({"Minutes after meal":r[f"minute{m}"],"Glucose (mg/dL)":r[f"observed{m}"]} for m in READING_MINUTES if r.get(f"observed{m}") is not None)
        figure=go.Figure(go.Scatter(x=[p["Minutes after meal"] for p in points],y=[p["Glucose (mg/dL)"] for p in points],mode="markers",marker={"size":10,"color":"#147D73"},name="Measured readings"))
        figure.update_layout(height=300,xaxis_title="Actual minutes after meal start",yaxis_title="Glucose (mg/dL)",xaxis={"tickvals":[0,15,30,45,60,120]})
        st.plotly_chart(figure,width="stretch",config={"displayModeBar":False})
        st.caption("Only entered measurements are plotted. Missing timepoints are not filled or inferred.")
    if st.session_state.get("outcome_alert"):
        st.warning(st.session_state["outcome_alert"])
    if measured:
        frame=pd.DataFrame({"Meal time":[parse_time(r["eaten_at"]).astimezone(LOCAL_TZ) for r in measured],
                            "Measured 2h":[r["observed120"] for r in measured],"Estimate saved before outcome":[r["pred120"] for r in measured]})
        st.line_chart(frame,x="Meal time",y=["Measured 2h","Estimate saved before outcome"],color=["#147D73","#B99867"])
        # Prequential error, not a score from refitting the same labels.
        error=np.mean([abs(r["observed120"]-r["pred120"]) for r in measured])
        st.caption(f"Error of estimates saved before their outcomes: {error:.1f} mg/dL across {len(measured)} meals. This is descriptive and includes excluded outcomes.")
    if rows:
        table=history_table(rows)
        st.caption("History and CSV exports use mg/dL. Actual elapsed minutes are included for every saved follow-up reading.")
        st.dataframe(table,hide_index=True,width="stretch")
        st.download_button("Export my meal history",safe_csv(table),"meal_history.csv","text/csv")
    with st.expander("Personal learning details"):
        st.json(learning_summary(personal))
        st.caption("The model needs at least 12 eligible meals and four later validation meals. An update activates only if it lowers error by at least 5% and 1 mg/dL. Backfilled outcomes cannot count as knowledge available before an earlier prediction.")


def custom_page(store,profile):
    st.title("Add a food you eat")
    st.write("Save a nutrition label or a recipe calculation. It becomes available immediately; its glucose response is learned only after measured meals.")
    if profile["id"]=="demo":
        st.info("Create a personal profile to add foods. Custom foods stay within that profile.")
        return
    with st.form("new_food_"+profile["id"]):
        name=st.text_input("Food name",max_chars=100,placeholder="For example, my vegetable ragi dosa")
        st.markdown("**Nutrition per 100 g of the food as eaten**")
        st.caption("Do not enter per-serving values here. Use the label or a recipe calculation; do not guess clinical properties such as glycemic index.")
        cols=st.columns(3)
        labels={"energy_kcal":"Energy (kcal)","carb_g":"Carbohydrate (g)","protein_g":"Protein (g)","fat_g":"Fat (g)","fibre_g":"Fibre (g)","sugar_g":"Free / added sugar (g)"}
        nutrients={}
        for i,key in enumerate(NUTRIENTS):
            nutrients[key]=cols[i%3].number_input(labels[key],min_value=0.,max_value=950. if key=="energy_kcal" else 100.,value=None,key="new_"+key,placeholder="Per 100 g")
        serving=st.number_input("Your usual weighed serving (g)",5.,750.,100.,step=5.)
        note=st.text_area("Nutrition source / recipe notes",max_chars=1000,placeholder="Brand label or ingredients, weights and cooked yield")
        confirmed=st.checkbox("These values describe the food as eaten, per 100 g; zero values are known zeros.")
        if st.form_submit_button("Add to my foods",type="primary"):
            try:
                if not confirmed or any(v is None for v in nutrients.values()):
                    raise ValueError("Fill every nutrient value and confirm the 100 g basis. Unknown values must not be silently treated as zero.")
                store.add_food(profile["id"],name,nutrients,serving,note)
                toast_result("Food added. You can now select it in Meal check and build evidence with measured outcomes.")
            except ValueError as exc:
                st.error(str(exc))
    foods=store.custom_foods(profile["id"])
    if foods:
        st.subheader("Your added foods")
        st.dataframe(pd.DataFrame(foods)[["name","serving_g",*NUTRIENTS,"notes"]],hide_index=True,width="stretch")


def research_page(profile):
    st.title("What the twin knows")
    st.write("Food composition, simulated response models and real glucose evidence are tracked separately.")
    audit=json.loads((REPORTS/"data_audit.json").read_text(encoding="utf-8"))
    synthetic=json.loads((REPORTS/"synthetic_metrics.json").read_text(encoding="utf-8"))
    observed=json.loads((REPORTS/"observed_metrics.json").read_text(encoding="utf-8"))
    a,b,c=st.columns(3)
    a.metric("Distinct foods",f"{audit['catalog_unique_foods']:,}")
    b.metric("Simulated meals",f"{audit['synthetic_rows']:,}")
    c.metric("Observed glucose windows",observed["rows"])
    st.subheader("Meal model · synthetic development")
    st.write(f"Selected model: **{synthetic['selected_model']}**. It estimates 60- and 120-minute glucose from the premeal reading, meal nutrients, recent activity and sleep.")
    st.dataframe(pd.DataFrame(synthetic["validation_leaderboard"])[["model","mae","mae_60","mae_120","training_seconds"]],hide_index=True,width="stretch")
    st.info(f"Held-out simulator MAE: {synthetic['test']['mae']:.2f} mg/dL. This is simulator imitation accuracy. Clinical accuracy is not established.")
    st.subheader("Observed glucose · separate benchmark")
    st.write(f"{observed['rows']} usable meal windows from {observed['participants']} Type 1 participants. The model forecasts after readings at 0, 15, 40 and 45 minutes; it does not establish which Indian food caused the response.")
    a,b,c=st.columns(3)
    a.metric("Observed holdout MAE",f"{observed['test']['mae']:.2f} mg/dL")
    b.metric("Persistence baseline MAE",f"{observed['test_persistence']['mae']:.2f} mg/dL")
    c.metric("Two-hour forecast MAE",f"{observed['test']['mae_120']:.2f} mg/dL")
    st.warning("The observed benchmark provides very limited support: overall performance is almost identical to persistence, and the two-hour forecast is worse. This is a research prototype, not a clinically validated digital twin.")
    with st.expander("Explore the separate 45-minute glucose forecast"):
        st.caption("Use readings from the same meal. Values here are mg/dL. This exploratory result does not change food lists or personal learning.")
        cols=st.columns(4)
        readings=[col.number_input(f"At {m} minutes",20.,600.,float(v),key=f"landmark_{m}") for col,m,v in zip(cols,[0,15,40,45],[110,125,145,150])]
        if st.button("Forecast from these four readings"):
            alert=next((safety_message(x) for x in readings if safety_message(x)),"")
            if alert:
                st.warning(alert)
            else:
                model=joblib.load(MODELS/"observed_forecast.joblib")
                pred=model["model"].predict([readings])[0]
                q=model["q90"]
                st.dataframe(pd.DataFrame({"Minute":[60,120],"Forecast (mg/dL)":pred.round(1),"Exploratory low":np.maximum(20,pred-q).round(1),"Exploratory high":np.minimum(600,pred+q).round(1)}),hide_index=True)
                st.caption("Ranges come from small-sample grouped cross-validation errors and have no proven clinical coverage.")
    with st.expander("Data sources and bias audit",expanded=True):
        st.write(f"Original workbook preserved: {audit['original_rows']:,} rows. Full INDB source: {audit['source_rows']:,} rows, including {audit['source_names_absent_from_original']:,} names absent from the original. {audit['quarantined_rows']} invalid source rows quarantined. {audit['calculated_portion_rows']:,} calculated portions are not extra measured foods.")
        st.markdown("[Indian Nutrient Databank and methodology](https://www.anuvaad.org.in/indian-nutrient-databank/) · [ICMR-NIN IFCT 2017](https://www.nin.res.in/ebooks/IFCT2017_16122024.pdf) · [D1NAMO source dataset](https://doi.org/10.5281/zenodo.5651217) · [NIDDK glucose targets](https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes)")
        st.write("Remaining gaps include Indian patient outcomes, treatment and insulin context, regional cooking differences, portion error, age and pregnancy, kidney disease and other conditions. More simulated rows do not remove those gaps.")
        st.caption("No patient or meal records are transmitted by the app. Public-source download is a separate preparation step. The local database is not encrypted and profiles are not an authentication boundary.")
    report=ROOT/"MODEL_REPORT.md"
    if report.exists():
        st.download_button("Download full model and process report",report.read_bytes(),file_name="MODEL_REPORT.md",mime="text/markdown")


def main():
    st.set_page_config(page_title="Glucose Twin",page_icon="🌿",layout="wide",initial_sidebar_state="expanded")
    st.markdown("""<style>
        .block-container {max-width:1200px;padding-top:2.2rem;padding-bottom:3rem}
        h1 {font-size:2.45rem!important;font-weight:650!important;letter-spacing:-.055rem}
        h2,h3 {letter-spacing:-.02rem}
        [data-testid="stMetric"] {background:white;border:1px solid #DFE9E3;border-radius:12px;padding:16px}
        [data-testid="stSidebar"] {border-right:1px solid #DDE7E1}
        div.stButton>button {min-height:2.75rem;border-radius:9px}
        [data-testid="stCaptionContainer"] {color:#4C6860}
        </style>""",unsafe_allow_html=True)
    if not (MODELS/"meal_prior.joblib").exists():
        st.error("The trained model is missing. Run the three setup commands in README.md, then restart the app.")
        st.stop()
    store=Store(os.environ.get("GLUCOSE_TWIN_DB") or None)
    profile,page,context=sidebar(store)
    # Clear profile-scoped transient output on switching; never carry another profile's alert or estimate.
    if st.session_state.get("active_profile")!=profile["id"]:
        for key in ["estimate","outcome_alert","personal","learning_token"]:
            st.session_state.pop(key,None)
        st.session_state["active_profile"]=profile["id"]
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    if profile["restrictions"]:
        st.info("Your care notes: "+profile["restrictions"])
    alert=safety_message(context["baseline"])
    if alert:
        st.error(alert)
    artifact=artifacts((MODELS/"meal_prior.joblib").stat().st_mtime_ns)
    rows=[] if profile["id"]=="demo" else store.meals(profile["id"])
    custom=[] if profile["id"]=="demo" else store.custom_foods(profile["id"])
    catalog=load_catalog(custom)
    personal=learning_state(rows,artifact,profile)
    if page=="Meal check":
        meal_page(store,profile,context,catalog,artifact,personal,rows)
    elif page=="Log a meal":
        meal_page(store,profile,context,catalog,artifact,personal,rows,log_only=True)
    elif page=="My food lists":
        foods_page(profile,context,catalog,artifact,personal,rows)
    elif page=="Meal journal":
        journal_page(store,profile,rows,artifact,personal)
    elif page=="Add a food":
        custom_page(store,profile)
    else:
        research_page(profile)

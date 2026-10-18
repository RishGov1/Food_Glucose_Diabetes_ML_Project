from datetime import datetime, timedelta, timezone
import json

import pytest
from streamlit.testing.v1 import AppTest
from twin.config import ROOT


@pytest.fixture
def app(tmp_path,monkeypatch):
    monkeypatch.setenv("GLUCOSE_TWIN_DB",str(tmp_path/"ui.sqlite3"))
    return AppTest.from_file(str(ROOT/"app.py"),default_timeout=60).run()


def by_label(elements,label):
    return next(x for x in elements if x.label==label)


def test_demo_meal_and_stale_result(app):
    assert not app.exception
    by_label(app.button,"Check this meal").click().run()
    assert not app.exception
    assert any(x.label=="Illustrative two-hour glucose" for x in app.metric)
    by_label(app.number_input,"Before-meal glucose").set_value(145.).run()
    assert not any(x.label=="Illustrative two-hour glucose" for x in app.metric)


def test_every_page_renders(app):
    for page in ["Log a meal","My food lists","Meal journal","Add a food","Model & data","Meal check"]:
        app.radio(key="page").set_value(page).run(timeout=90)
        assert not app.exception, page


def test_profile_custom_food_and_isolation(app):
    by_label(app.text_input,"Profile nickname").set_value("Test person")
    by_label(app.button,"Create profile").click().run()
    assert not app.exception
    app.radio(key="page").set_value("Add a food").run()
    by_label(app.text_input,"Food name").set_value("Test dal bowl")
    for label,value in [("Energy (kcal)",100.),("Carbohydrate (g)",15.),("Protein (g)",5.),("Fat (g)",2.),("Fibre (g)",3.),("Free / added sugar (g)",0.)]:
        by_label(app.number_input,label).set_value(value)
    by_label(app.checkbox,"These values describe the food as eaten, per 100 g; zero values are known zeros.").check()
    by_label(app.button,"Add to my foods").click().run()
    assert not app.exception
    assert any("Test dal bowl" in str(x.value) for x in app.dataframe)
    app.selectbox(key="profile_select").set_value("demo").run()
    assert not app.exception
    assert not app.dataframe


def test_low_glucose_pauses_verdict(app):
    by_label(app.number_input,"Before-meal glucose").set_value(50.).run()
    by_label(app.button,"Check this meal").click().run()
    assert not app.exception
    assert any("Very low" in e.value for e in app.error)
    assert not any(x.label=="Illustrative two-hour glucose" for x in app.metric)


def test_save_meal_and_measured_outcome(app):
    by_label(app.text_input,"Profile nickname").set_value("Journal test")
    by_label(app.button,"Create profile").click().run()
    by_label(app.button,"Check this meal").click().run()
    local=timezone(timedelta(hours=5,minutes=30))
    yesterday=(datetime.now(local)-timedelta(days=1)).date()
    by_label(app.date_input,"Meal date").set_value(yesterday)
    by_label(app.checkbox,"I ate this portion and entered my measured premeal glucose.").check()
    by_label(app.button,"Save meal to journal").click().run()
    assert not app.exception
    app.radio(key="page").set_value("Meal journal").run()
    by_label(app.number_input,"120-minute glucose (2 hours)").set_value(145.)
    by_label(app.number_input,"60-minute glucose (1 hour)").set_value(155.)
    by_label(app.checkbox,"These are actual meter / CGM readings, not predicted or demonstration values.").check()
    by_label(app.button,"Save readings and update twin").click().run()
    assert not app.exception
    assert any(x.label=="Measured outcomes" and x.value=="1" for x in app.metric)
    assert by_label(app.number_input,"120-minute glucose (2 hours)").disabled
    assert not by_label(app.number_input,"15-minute glucose").disabled


def test_direct_log_and_incremental_readings(app):
    by_label(app.text_input,"Profile nickname").set_value("Incremental readings")
    by_label(app.button,"Create profile").click().run()
    app.radio(key="page").set_value("Log a meal").run()
    assert not any(x.label=="Check this meal" for x in app.button)
    foods=by_label(app.multiselect,"What is in the meal?")
    # The same meal can contain multiple foods.
    from twin.service import load_catalog
    catalog=load_catalog()
    additional=catalog[catalog.meal_eligible & catalog.name.str.contains("sambar",case=False)].iloc[0].food_id
    foods.set_value(foods.value+[additional]).run()
    local=timezone(timedelta(hours=5,minutes=30))
    by_label(app.date_input,"Meal date").set_value((datetime.now(local)-timedelta(days=1)).date())
    by_label(app.checkbox,"I ate this portion and entered my measured premeal glucose.").check()
    by_label(app.button,"Save meal to journal").click().run()
    assert app.radio(key="page").value=="Meal journal"
    by_label(app.number_input,"15-minute glucose").set_value(125.)
    by_label(app.checkbox,"These are actual meter / CGM readings, not predicted or demonstration values.").check()
    by_label(app.button,"Save readings and update twin").click().run()
    assert not app.exception
    assert by_label(app.number_input,"15-minute glucose").value==125.
    assert by_label(app.number_input,"15-minute glucose").disabled
    assert by_label(app.number_input,"120-minute glucose (2 hours)").value is None
    for label,value in [("30-minute glucose",140.),("45-minute glucose",155.),("60-minute glucose (1 hour)",150.),("120-minute glucose (2 hours)",130.)]:
        by_label(app.number_input,label).set_value(value)
    by_label(app.checkbox,"These are actual meter / CGM readings, not predicted or demonstration values.").check()
    by_label(app.button,"Save readings and update twin").click().run()
    assert not app.exception
    assert any("All five follow-up readings" in x.value for x in app.success)
    table=next(x.value for x in app.dataframe if "Measured 15" in x.value.columns)
    assert table.iloc[0]["Measured 15"]==125.
    assert table.iloc[0]["Measured 120"]==130.


def test_mmol_display_and_prediction(app):
    units=next(x for x in app.selectbox if x.label=="Glucose units")
    units.set_value("mmol/L").run()
    by_label(app.number_input,"Before-meal glucose").set_value(6.2).run()
    by_label(app.button,"Check this meal").click().run()
    assert not app.exception
    assert any(x.label=="Illustrative two-hour glucose" for x in app.metric)
    by_label(app.number_input,"Before-meal glucose").set_value(33.29).run()
    assert not app.exception
    assert any("High glucose" in x.value for x in app.error)

import sqlite3
from contextlib import closing
import numpy as np
from streamlit.testing.v1 import AppTest
import src.personal as personal
from src.forecast import ROOT


def by_label(elements, label):
    return next(item for item in elements if item.label == label)


def test_forecast_save_update_and_demo_isolation(tmp_path, monkeypatch):
    original = personal.connect
    database = tmp_path / "dashboard.sqlite3"
    monkeypatch.setattr(personal, "connect", lambda path=None: original(database))
    at = AppTest.from_file(ROOT / "app.py", default_timeout=45).run()
    assert not at.exception
    by_label(at.button, "Load test example").click().run()
    assert not at.exception
    assert at.number_input(key="early_40").value == 130
    by_label(at.button, "Forecast next readings").click().run()
    assert not at.exception
    before = at.session_state["meal_forecast"]["forecast"]
    assert len(before) == 2 and np.isfinite(before).all()
    by_label(at.checkbox, "These are measured outcomes (or synthetic values in Demo mode)").check()
    by_label(at.button, "Save outcome & update my twin").click().run()
    assert not at.exception
    assert any("Twin updated from 1" in item.value for item in at.success)
    by_label(at.button, "Save outcome & update my twin").click().run()
    assert not at.exception
    assert any("no duplicate" in item.value for item in at.success)
    by_label(at.button, "Forecast next readings").click().run()
    after = at.session_state["meal_forecast"]["forecast"]
    assert not np.allclose(before, after)
    by_label(at.toggle, "Demo mode").set_value(False).run()
    assert not at.exception
    by_label(at.button, "Forecast next readings").click().run()
    np.testing.assert_allclose(at.session_state["meal_forecast"]["forecast"], before)
    with closing(sqlite3.connect(database)) as db:
        assert db.execute("SELECT COUNT(*) FROM meals").fetchone()[0] == 1


def test_stale_forecast_cannot_save(tmp_path, monkeypatch):
    original = personal.connect
    monkeypatch.setattr(personal, "connect", lambda path=None: original(tmp_path / "stale.db"))
    at = AppTest.from_file(ROOT / "app.py", default_timeout=45).run()
    by_label(at.button, "Load test example").click().run()
    by_label(at.button, "Forecast next readings").click().run()
    at.number_input(key="early_40").set_value(150.).run()
    assert not at.exception
    assert not any(b.label == "Save outcome & update my twin" for b in at.button)


def test_npu_comparison_and_food_search(tmp_path, monkeypatch):
    original = personal.connect
    monkeypatch.setattr(personal, "connect", lambda path=None: original(tmp_path / "npu.db"))
    at = AppTest.from_file(ROOT / "app.py", default_timeout=45).run()
    by_label(at.button, "Load test example").click().run()
    by_label(at.button, "Forecast next readings").click().run()
    by_label(at.selectbox, "Neural inference device").set_value("NPU")
    by_label(at.button, "Run neural comparison").click().run()
    assert not at.exception and not at.error
    assert any("NPU: 60 min" in item.value for item in at.markdown)
    by_label(at.text_input, "Search foods").set_value("cake").run()
    assert not at.exception
    assert any("Food" in table.value.columns and table.value["Food"].str.contains("cake", case=False).any() for table in at.dataframe)

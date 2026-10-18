"""Live browser verification on an isolated database and port; never touches personal history."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime,timedelta,timezone
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import requests
from playwright.sync_api import sync_playwright,expect
from twin.config import REPORTS,ROOT


def main():
    chrome=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    port=8502
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1",port))==0:
            raise RuntimeError("Port 8502 is already in use; refusing to interrupt another app.")
    with tempfile.TemporaryDirectory(prefix="glucose_twin_browser_") as temp:
        env=dict(os.environ,GLUCOSE_TWIN_DB=str(Path(temp)/"browser.sqlite3"))
        log=open(REPORTS/"browser_server.log","w",encoding="utf-8")
        proc=subprocess.Popen([sys.executable,"-m","streamlit","run",str(ROOT/"app.py"),"--server.port",str(port),"--server.address","127.0.0.1","--server.headless","true"],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        try:
            for _ in range(60):
                try:
                    if requests.get(f"http://127.0.0.1:{port}/_stcore/health",timeout=1).status_code==200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.5)
            with sync_playwright() as pw:
                browser=pw.chromium.launch(executable_path=str(chrome) if chrome.exists() else None,headless=True)
                page=browser.new_page(viewport={"width":1440,"height":1100},device_scale_factor=1)
                errors=[]
                page.on("pageerror",lambda e:errors.append(str(e)))
                page.goto(f"http://127.0.0.1:{port}",wait_until="networkidle")
                expect(page.get_by_role("heading",name="A clearer picture of your next meal")).to_be_visible(timeout=30000)
                page.get_by_role("button",name="Check this meal",exact=True).click()
                expect(page.get_by_text("Illustrative two-hour glucose",exact=True)).to_be_visible(timeout=15000)
                expect(page.get_by_test_id("stPlotlyChart").locator(".js-plotly-plot")).to_be_visible(timeout=15000)
                page.screenshot(path=str(REPORTS/"app_meal_desktop.png"),full_page=True)
                page.get_by_text("My food lists",exact=True).click()
                expect(page.get_by_role("heading",name="Your food lists")).to_be_visible()
                search=page.get_by_role("textbox",name="Search Indian foods")
                search.fill("paneer")
                search.press("Enter")
                page.wait_for_timeout(1500)
                page.screenshot(path=str(REPORTS/"app_foods_desktop.png"),full_page=True)
                page.get_by_text("Create a personal profile",exact=True).click()
                page.get_by_role("textbox",name="Profile nickname").fill("Browser test profile")
                page.get_by_role("button",name="Create profile",exact=True).click()
                expect(page.get_by_text("Your local profile is ready.",exact=True)).to_be_visible(timeout=15000)
                page.get_by_text("Log a meal",exact=True).click()
                expect(page.get_by_role("heading",name="Save a meal you ate")).to_be_visible()
                yesterday=datetime.now(timezone(timedelta(hours=5,minutes=30)))-timedelta(days=1)
                date_input=page.get_by_label("Meal date",exact=True)
                for part,value in [("year",yesterday.year),("month",yesterday.month),("day",yesterday.day)]:
                    segment=date_input.get_by_role("spinbutton",name=part,exact=False)
                    segment.fill(str(value))
                    segment.press("Tab")
                page.get_by_text("I ate this portion and entered my measured premeal glucose.",exact=True).click()
                expect(page.get_by_role("checkbox",name="I ate this portion and entered my measured premeal glucose.")).to_be_checked()
                page.get_by_role("button",name="Save meal to journal",exact=True).click()
                expect(page.get_by_text("Meal saved.",exact=False)).to_be_visible(timeout=15000)
                page.get_by_text("Meal journal",exact=True).click()
                page.get_by_label("15-minute glucose",exact=True).fill("125")
                page.get_by_text("These are actual meter / CGM readings, not predicted or demonstration values.",exact=True).click()
                page.get_by_role("button",name="Save readings and update twin",exact=True).click()
                expect(page.get_by_label("15-minute glucose",exact=True)).to_be_disabled(timeout=15000)
                for label,value in [("30-minute glucose","140"),("45-minute glucose","155"),("60-minute glucose (1 hour)","150"),("120-minute glucose (2 hours)","130")]:
                    page.get_by_label(label,exact=True).fill(value)
                confirmation=page.get_by_role("checkbox",name="These are actual meter / CGM readings, not predicted or demonstration values.")
                if not confirmation.is_checked():
                    page.get_by_text("These are actual meter / CGM readings, not predicted or demonstration values.",exact=True).click()
                page.get_by_role("button",name="Save readings and update twin",exact=True).click()
                expect(page.get_by_text("All five follow-up readings are saved for every logged meal.",exact=True)).to_be_visible(timeout=15000)
                page.screenshot(path=str(REPORTS/"app_journal_desktop.png"),full_page=True)
                page.get_by_text("Model & data",exact=True).click()
                expect(page.get_by_role("heading",name="What the twin knows")).to_be_visible()
                page.screenshot(path=str(REPORTS/"app_models_desktop.png"),full_page=True)
                mobile=browser.new_page(viewport={"width":390,"height":844},is_mobile=True,has_touch=True,device_scale_factor=1)
                mobile.goto(f"http://127.0.0.1:{port}",wait_until="networkidle")
                expect(mobile.get_by_role("heading",name="A clearer picture of your next meal")).to_be_visible(timeout=30000)
                # Streamlit starts with an expanded sidebar; collapse it for the meal view.
                collapse=mobile.get_by_test_id("stSidebarCollapseButton").get_by_role("button")
                if collapse.count() and collapse.is_visible():
                    collapse.click()
                mobile.get_by_role("button",name="Check this meal",exact=True).click()
                expect(mobile.get_by_text("Illustrative two-hour glucose",exact=True)).to_be_visible(timeout=15000)
                expect(mobile.get_by_test_id("stPlotlyChart").locator(".js-plotly-plot")).to_be_visible(timeout=15000)
                mobile.get_by_text("Illustrative two-hour glucose",exact=True).scroll_into_view_if_needed()
                mobile.screenshot(path=str(REPORTS/"app_meal_mobile.png"),full_page=True)
                overflow=mobile.evaluate("document.documentElement.scrollWidth > window.innerWidth + 4")
                assert not overflow,"Page overflows the mobile viewport"
                assert not errors,errors
                assert page.get_by_test_id("stException").count()==0
                browser.close()
                result={"desktop":"1440x1100","mobile":"390x844","workflow":"demo estimate, food search, isolated profile, direct meal log, incremental 15/30/45/60/120-minute readings, learning check, model report",
                        "browser_errors":errors,"mobile_horizontal_overflow":overflow,"database":"isolated temporary SQLite; deleted after run","result":"passed"}
                (REPORTS/"browser_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
                print(json.dumps(result,indent=2))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            log.close()


if __name__=="__main__":
    main()

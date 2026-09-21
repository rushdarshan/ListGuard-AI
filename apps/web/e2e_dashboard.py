"""E2E: reviewer-dashboard flow (Playwright + real API on SQLite).

Bootstraps everything in-process: file-SQLite API (uvicorn thread, dashboard
HTML mounted same-origin at /web) + seeds via HTTP + drives Chromium.
No new dependencies: fastapi/uvicorn/httpx/playwright already installed.

Run:  python apps/web/e2e_dashboard.py   (from repo root)
Artifacts: apps/web/test-results/ (screenshots on failure, videos, traces)
"""
import os
import sys
import tempfile
import threading
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "apps", "api"))

TMP = tempfile.mkdtemp(prefix="e2e-lg-")
os.environ["DATABASE_URL"] = f"sqlite:///{TMP}/e2e.db"
os.environ["REDIS_URL"] = ""  # redis skipped in /ready

from fastapi.staticfiles import StaticFiles  # noqa: E402

from app import models  # noqa: E402,F401 - register tables
from app.core.db import Base, engine  # noqa: E402
from app.main import create_app  # noqa: E402

API_PORT = 8931
API = f"http://127.0.0.1:{API_PORT}"
WEB_DIR = os.path.join(REPO, "apps", "web")
RESULTS = os.path.join(WEB_DIR, "test-results")
os.makedirs(RESULTS, exist_ok=True)

# dashboard.html imports ./dashboard.ts (typed) — browsers can't parse TS,
# so serve the 3 pure helpers as plain JS. Test-only shim, source untouched.
TS_SHIM = """export function initialState(){return {status:"loading",payload:null,error:null};}
export function errorState(m){return {status:"error",payload:null,error:m};}
export function applyReviewDecision(h,e){if(!e.id)throw new Error("review event id is required");
if(!["accepted","rejected","corrected"].includes(e.decision))throw new Error("unknown decision: "+e.decision);
return [...h,e];}"""

SHA = "b" * 64


def seed(http, tag):
    """Listing + image + attr/dup/price predictions. Returns listing id."""
    lid = http.post("/v1/listings", json={
        "title": "Sony A7III", "description": "Mirrorless camera body",
        "seller_id": "seller-1"}).json()["id"]
    http.post(f"/v1/listings/{lid}/images", json={
        "storage_key": "s3://b/img.jpg", "filename": "img.jpg",
        "content_type": "image/jpeg", "byte_size": 5000,
        "width": 800, "height": 600, "sha256": SHA})
    mv = http.post("/v1/model-versions",
                   json={"name": f"attr-{tag}", "version": "1.0.0",
                         "config": {}}).json()
    http.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "prediction_type": "attribute_extraction",
        "attributes": [{"key": "brand", "value": "sony", "confidence": 0.9,
                        "evidence": [{"source": "text", "text_span":
                                      {"field": "title", "start": 0, "end": 4}}]}]})
    http.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "prediction_type": "duplicate_detection",
        "attributes": [{"key": "dup:x", "value": "other-id", "confidence": 0.82}]})
    http.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "prediction_type": "price_estimation",
        "attributes": [{"key": "low", "value": "100", "confidence": 0.7},
                       {"key": "high", "value": "150", "confidence": 0.7}]})
    return lid


def start_api():
    import uvicorn
    Base.metadata.create_all(bind=engine)
    app = create_app()
    app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web")
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=API_PORT,
                                        log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    import httpx
    for _ in range(100):
        try:
            if httpx.get(f"{API}/health", timeout=1).status_code == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("API did not start")


def run():
    import httpx
    from playwright.sync_api import sync_playwright

    start_api()
    http = httpx.Client(base_url=API, timeout=10)
    lid_accept = seed(http, "a")
    lid_correct = seed(http, "b")

    results = []

    def load_dashboard(page, lid):
        page.goto(f"{API}/web/dashboard.html")
        page.fill("#listing-id", lid)
        page.click("#load")
        page.locator("#d-title").wait_for(state="visible", timeout=5000)

    def check(name, fn):
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(record_video_dir=RESULTS)
            page = ctx.new_page()
            page.route("**/dashboard.ts", lambda r: r.fulfill(
                content_type="text/javascript", body=TS_SHIM))
            ctx.tracing.start(screenshots=True, snapshots=True)
            try:
                fn(page)
                results.append((name, True, ""))
                print(f"PASS: {name}")
            except Exception as e:  # noqa: BLE001
                shot = os.path.join(RESULTS, f"{name}.png")
                page.screenshot(path=shot)
                ctx.tracing.stop(path=os.path.join(RESULTS, f"{name}.zip"))
                results.append((name, False, f"{e}\nScreenshot: {shot}"))
                print(f"FAIL: {name}: {e}")
                traceback.print_exc()
            finally:
                browser.close()

    def t_accept(page):
        load_dashboard(page, lid_accept)
        assert page.locator("#d-title").inner_text() == "Sony A7III"
        assert "img.jpg" in page.locator("#d-images").inner_text()
        assert "sony" in page.locator("#d-attrs").inner_text()
        assert "100" in page.locator("#d-price").inner_text()
        page.fill("#r-reviewer", "rev-e2e")
        page.click('button[data-decision="accepted"]')
        page.get_by_text("Recorded accepted.").wait_for(timeout=5000)
        hist = page.locator("#d-history").inner_text()
        assert "accepted by rev-e2e" in hist, hist

    def t_correct_append_only(page):
        load_dashboard(page, lid_correct)
        page.fill("#r-reviewer", "rev-e2e")
        page.click('button[data-decision="accepted"]')
        page.get_by_text("Recorded accepted.").wait_for(timeout=5000)
        page.fill("#r-corrected", '{"brand":"canon"}')
        page.click('button[data-decision="corrected"]')
        page.get_by_text("Recorded corrected.").wait_for(timeout=5000)
        items = page.locator("#d-history li").all_inner_texts()
        assert len(items) == 2, items  # append-only: both events kept
        assert "canon" in items[1]
        # original prediction untouched — reload shows same extracted value
        load_dashboard(page, lid_correct)
        assert "sony" in page.locator("#d-attrs").inner_text()

    def t_edges(page):
        page.goto(f"{API}/web/dashboard.html")
        page.fill("#listing-id", "00000000-0000-4000-8000-000000000000")
        page.click("#load")
        page.locator("#error:not([hidden])").wait_for(timeout=5000)
        assert "404" in page.locator("#error").inner_text()
        load_dashboard(page, lid_accept)
        page.fill("#r-reviewer", "rev-e2e")
        page.fill("#r-corrected", "not-json{")
        page.click('button[data-decision="corrected"]')
        page.locator("#error:not([hidden])").wait_for(timeout=5000)
        assert "valid JSON" in page.locator("#error").inner_text()

    check("load-and-accept", t_accept)
    check("correct-append-only", t_correct_append_only)
    check("edge-cases", t_edges)

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nE2E Test Results\n================\n"
          f"PASS: Passed: {passed}\nFAIL: Failed: {len(results) - passed}\n"
          f"SKIPPED: Skipped: 0")
    for name, ok, err in results:
        if not ok:
            print(f"- {name}: {err.splitlines()[0]}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    run()

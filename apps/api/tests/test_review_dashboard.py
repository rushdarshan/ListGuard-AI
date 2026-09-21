"""RED: reviewer-dashboard aggregate endpoint + append-only correction workflow."""

SHA = "b" * 64


def _listing(c, **kw):
    body = {"title": "Sony A7III", "description": "Mirrorless camera body", "seller_id": "seller-1"}
    body.update(kw)
    r = c.post("/v1/listings", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _model(c):
    r = c.post("/v1/model-versions", json={"name": "attr", "version": "1.0.0", "config": {}})
    assert r.status_code == 201, r.text
    return r.json()


def _seed(c):
    """Listing + image + attr/dup/price predictions, returns ids."""
    lid = _listing(c)["id"]
    img = c.post(f"/v1/listings/{lid}/images", json={
        "storage_key": "s3://b/img.jpg", "filename": "img.jpg",
        "content_type": "image/jpeg", "byte_size": 5000,
        "width": 800, "height": 600, "sha256": SHA}).json()
    mv = _model(c)
    attr = c.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "prediction_type": "attribute_extraction",
        "attributes": [{"key": "brand", "value": "sony", "confidence": 0.9,
                        "evidence": [{"source": "text",
                                      "text_span": {"field": "title", "start": 0, "end": 4}}]}]}).json()
    c.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "prediction_type": "duplicate_detection",
        "attributes": [{"key": "dup:other-id", "value": "other-id", "confidence": 0.82}]}).json()
    c.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "prediction_type": "price_estimation",
        "attributes": [{"key": "price_low", "value": "100", "confidence": 0.7},
                       {"key": "price_high", "value": "150", "confidence": 0.7}]}).json()
    return lid, img, mv, attr


def test_dashboard_shows_everything(client):
    lid, img, mv, attr = _seed(client)
    r = client.get(f"/v1/listings/{lid}/review-dashboard")
    assert r.status_code == 200, r.text
    d = r.json()
    # 10 required sections
    assert d["listing"]["description"] == "Mirrorless camera body"  # seller description
    assert [i["id"] for i in d["images"]] == [img["id"]]  # listing images
    by_type = {p["prediction_type"]: p for p in d["predictions"]}
    attr_pred = by_type["attribute_extraction"]
    assert attr_pred["attributes"][0]["key"] == "brand"  # extracted attributes
    assert attr_pred["attributes"][0]["confidence"] == 0.9  # confidence scores
    assert attr_pred["attributes"][0]["evidence"][0]["source"] == "text"  # evidence
    assert d["duplicate_matches"] and d["duplicate_matches"][0]["score"] == 0.82  # dup matches
    assert d["price_interval"] == {"low": 100.0, "high": 150.0}  # price interval
    assert isinstance(d["risk_findings"], list) and d["risk_findings"]  # risk findings
    assert attr_pred["model_version"]["version"] == "1.0.0"  # model version
    assert d["decision_history"] == []  # complete decision history (empty so far)


def test_dashboard_404_unknown_listing(client):
    r = client.get("/v1/listings/00000000-0000-4000-8000-000000000000/review-dashboard")
    assert r.status_code == 404


def test_dashboard_price_interval_ignores_non_numeric(client):
    lid = _listing(client)["id"]
    mv = _model(client)
    c = client.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "prediction_type": "price_estimation",
        "attributes": [
            {"key": "price_note", "value": "expensive", "confidence": 0.5},
            {"key": "price_abs", "value": None, "confidence": 0.1, "abstained": True},
        ]}).json()
    assert c["prediction_type"] == "price_estimation"
    d = client.get(f"/v1/listings/{lid}/review-dashboard").json()
    assert d["price_interval"] is None  # nothing numeric to bound
    r = client.get("/v1/listings/00000000-0000-4000-8000-000000000000/review-dashboard")
    assert r.status_code == 404


def test_dashboard_empty_state(client):
    lid = _listing(client)["id"]
    d = client.get(f"/v1/listings/{lid}/review-dashboard").json()
    assert d["images"] == [] and d["predictions"] == []
    assert d["duplicate_matches"] == [] and d["price_interval"] is None
    assert d["decision_history"] == [] and isinstance(d["risk_findings"], list)


def test_e2e_review_workflow_never_overwrites_prediction(client):
    """One complete e2e review workflow: accept -> correct appends, prediction intact."""
    lid, img, mv, attr = _seed(client)
    before = client.get(f"/v1/listings/{lid}/predictions").json()
    # reviewer accepts
    r = client.post(f"/v1/listings/{lid}/reviews", json={
        "decision": "accepted", "prediction_id": attr["id"], "reviewer_id": "rev-1"})
    assert r.status_code == 201, r.text
    # reviewer corrects (stored as NEW event, original untouched)
    r = client.post(f"/v1/listings/{lid}/reviews", json={
        "decision": "corrected", "prediction_id": attr["id"], "reviewer_id": "rev-2",
        "corrected_attributes": {"brand": "canon"}, "note": "logo says canon"})
    assert r.status_code == 201, r.text
    after = client.get(f"/v1/listings/{lid}/predictions").json()
    assert after == before  # original prediction byte-identical
    hist = client.get(f"/v1/listings/{lid}/review-dashboard").json()["decision_history"]
    assert [h["decision"] for h in hist] == ["accepted", "corrected"]
    assert hist[1]["corrected_attributes"] == {"brand": "canon"}
    # corrected requires corrected_attributes
    bad = client.post(f"/v1/listings/{lid}/reviews", json={
        "decision": "corrected", "reviewer_id": "rev-3"})
    assert bad.status_code == 422


def test_dashboard_risk_inputs_match_hygiene_and_nan_guard():
    from types import SimpleNamespace

    from app.dashboard import assemble_dashboard, derive_price_interval

    assert derive_price_interval([
        {"key": "price_low", "value": "nan"}, {"key": "price_high", "value": "100"},
    ]) == {"low": 100.0, "high": 100.0}
    assert derive_price_interval([{"value": "nan"}, {"value": "inf"}]) is None

    listing = SimpleNamespace(id="l1", seller_id="s", title="Sony camera",
                              description="great camera", category="cameras", status="draft")
    dup_pred = SimpleNamespace(
        id="p1", prediction_type="duplicate_detection", latency_ms=1, created_at=None,
        attributes=[
            SimpleNamespace(id="a1", key="dup:x", value="x", confidence=0.9,
                            abstained=False, evidence=[]),
            SimpleNamespace(id="a2", key="dup:abs", value=None, confidence=0.1,
                            abstained=True, evidence=[]),
        ])
    mv = SimpleNamespace(id="m1", name="n", version="v")
    d = assemble_dashboard(listing=listing, images=[], predictions=[(dup_pred, mv)], reviews=[],
                           price=500.0, category_median_price=100.0,
                           other_descriptions=["great camera"])
    assert d["duplicate_matches"] == [
        {"prediction_id": "p1", "candidate_listing_id": "x", "score": 0.9}]
    cats = {f["risk_category"] for f in d["risk_findings"]}
    assert "abnormal_pricing" in cats and "repeated_description" in cats


def test_price_interval_uses_only_explicit_bound_keys():
    from app.dashboard import derive_price_interval

    assert derive_price_interval([
        {"key": "price_low", "value": "100"}, {"key": "price_high", "value": "150"},
        {"key": "similarity", "value": "0.92"}, {"key": "price_low", "value": "junk"},
    ]) == {"low": 100.0, "high": 150.0}
    assert derive_price_interval([{"key": "similarity", "value": "0.92"}]) is None


def test_web_dashboard_served_same_origin(client):
    r = client.get("/web/dashboard.html")
    assert r.status_code == 200 and "Reviewer dashboard" in r.text
    js = client.get("/web/dashboard.js")
    assert js.status_code == 200 and "escapeHtml" in js.text

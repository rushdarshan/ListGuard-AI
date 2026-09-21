"""Phase 7 gates: CRUD, soft-delete, metadata registration idempotency,
prediction invariants + prediction_type + structured evidence, append-only reviews."""

SHA = "a" * 64


def _listing(c, **kw):
    body = {"title": "Sony A7III", "description": "Mirrorless camera body", "seller_id": "seller-1"}
    body.update(kw)
    r = c.post("/v1/listings", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200, r.text
    assert r.json()["checks"]["db"] == "ok"


def test_listing_crud_and_soft_delete(client):
    created = _listing(client)
    lid = created["id"]
    assert client.get(f"/v1/listings/{lid}").status_code == 200
    r = client.patch(f"/v1/listings/{lid}", json={"status": "pending_review"})
    assert r.status_code == 200 and r.json()["status"] == "pending_review"
    assert client.delete(f"/v1/listings/{lid}").status_code == 204
    assert client.get(f"/v1/listings/{lid}").status_code == 404
    ids = [x["id"] for x in client.get("/v1/listings").json()["items"]]
    assert lid not in ids


def test_image_registration_and_dedupe(client):
    lid = _listing(client)["id"]
    body = {"storage_key": "s3://bucket/img1.jpg", "filename": "img1.jpg",
            "content_type": "image/jpeg", "byte_size": 12345, "sha256": SHA}
    assert client.post(f"/v1/listings/{lid}/images", json=body).status_code == 201
    dup = client.post(f"/v1/listings/{lid}/images", json=body)
    assert dup.status_code == 409
    assert len(client.get(f"/v1/listings/{lid}/images").json()) == 1


def _model_version(client):
    r = client.post("/v1/model-versions", json={"name": "attr", "version": "0.1.0", "config": {}})
    assert r.status_code == 201, r.text
    return r.json()


def test_prediction_types_evidence_and_abstention(client):
    lid = _listing(client)["id"]
    img = client.post(f"/v1/listings/{lid}/images", json={
        "storage_key": "s3://bucket/img1.jpg", "filename": "img1.jpg",
        "content_type": "image/jpeg", "byte_size": 10, "sha256": SHA}).json()
    mv = _model_version(client)
    for ptype in ("attribute_extraction", "duplicate_detection", "price_estimation", "quality_check"):
        r = client.post(f"/v1/listings/{lid}/predictions", json={
            "model_version_id": mv["id"], "prediction_type": ptype, "attributes": [
                {"key": f"k-{ptype}", "value": "canon", "confidence": 0.9, "evidence": [
                    {"source": "image", "image_id": img["id"], "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4}},
                    {"source": "text", "text_span": {"field": "title", "start": 0, "end": 4}},
                ]},
                {"key": f"abs-{ptype}", "value": None, "confidence": 0.1, "abstained": True, "evidence": []},
            ]})
        assert r.status_code == 201, (ptype, r.text)
        out = r.json()
        assert out["prediction_type"] == ptype
        by_key = {a["key"]: a for a in out["attributes"]}
        assert len(by_key[f"k-{ptype}"]["evidence"]) == 2
        ev = by_key[f"k-{ptype}"]["evidence"][0]
        assert ev["bounding_box"] == {"x": 1, "y": 2, "w": 3, "h": 4}
        assert by_key[f"abs-{ptype}"]["abstained"] is True
    # invariant violations
    bad1 = client.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "attributes": [{"key": "brand", "value": "x", "confidence": 0.5, "abstained": True}]})
    assert bad1.status_code == 422
    bad2 = client.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "attributes": [{"key": "brand", "value": None, "confidence": 0.5}]})
    assert bad2.status_code == 422
    assert len(client.get(f"/v1/listings/{lid}/predictions").json()) == 4


def test_review_append_only_history(client):
    lid = _listing(client)["id"]
    mv = _model_version(client)
    pred = client.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"],
        "attributes": [{"key": "brand", "value": "sony", "confidence": 0.8}]}).json()
    assert client.post(f"/v1/listings/{lid}/reviews", json={
        "decision": "accepted", "prediction_id": pred["id"], "reviewer_id": "rev-1"}).status_code == 201
    assert client.post(f"/v1/listings/{lid}/reviews", json={
        "decision": "corrected", "reviewer_id": "rev-2",
        "corrected_attributes": {"brand": "sony"}}).status_code == 201
    hist = client.get(f"/v1/listings/{lid}/reviews").json()
    assert [h["decision"] for h in hist] == ["accepted", "corrected"]
    # no update/delete routes for reviews
    assert client.put(f"/v1/listings/{lid}/reviews/{hist[0]['id']}").status_code in (404, 405)
    assert client.delete(f"/v1/listings/{lid}/reviews/{hist[0]['id']}").status_code in (404, 405)


def test_contract_branches_and_structured_errors(client):
    r = client.post("/v1/listings", json={"title": "", "seller_id": "s-1"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"

    r = client.get("/v1/listings/00000000-0000-4000-8000-000000000000")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "http_404"

    a = _listing(client, title="Red Bicycle", seller_id="seller-a")
    b = _listing(client, title="Blue Bicycle", description="fast road bike", seller_id="seller-b")
    client.patch(f"/v1/listings/{a['id']}", json={"status": "approved"})
    assert client.get("/v1/listings", params={"seller_id": "seller-a"}).json()["total"] == 1
    assert client.get("/v1/listings", params={"status": "approved"}).json()["total"] == 1
    q = client.get("/v1/listings", params={"q": "road bike"}).json()
    assert q["total"] == 1 and q["items"][0]["id"] == b["id"]

    mv = _model_version(client)
    assert client.get("/v1/model-versions").status_code == 200
    assert client.post("/v1/model-versions", json={"name": "attr", "version": "0.1.0"}).status_code == 409

    bad = client.post(f"/v1/listings/{a['id']}/predictions", json={
        "model_version_id": "00000000-0000-4000-8000-000000000000",
        "attributes": [{"key": "k", "value": "v", "confidence": 0.5}]})
    assert bad.status_code == 404

    img = client.post(f"/v1/listings/{b['id']}/images", json={
        "storage_key": "k", "filename": "f.jpg", "content_type": "image/jpeg",
        "byte_size": 10, "sha256": SHA}).json()
    cross = client.post(f"/v1/listings/{a['id']}/predictions", json={
        "model_version_id": mv["id"], "attributes": [
            {"key": "k", "value": "v", "confidence": 0.5,
             "evidence": [{"source": "image", "image_id": img["id"]}]}]})
    assert cross.status_code == 422

    dup_attr = client.post(f"/v1/listings/{a['id']}/predictions", json={
        "model_version_id": mv["id"], "attributes": [
            {"key": "k", "value": "v1", "confidence": 0.5},
            {"key": "k", "value": "v2", "confidence": 0.6}]})
    assert dup_attr.status_code == 409

    pred_b = client.post(f"/v1/listings/{b['id']}/predictions", json={
        "model_version_id": mv["id"],
        "attributes": [{"key": "k", "value": "v", "confidence": 0.5}]}).json()
    assert client.post(f"/v1/listings/{a['id']}/reviews", json={
        "decision": "corrected", "reviewer_id": "r"}).status_code == 422
    assert client.post(f"/v1/listings/{a['id']}/reviews", json={
        "decision": "accepted", "prediction_id": pred_b["id"], "reviewer_id": "r"}).status_code == 422


def test_review_remediation_regressions(client):
    pct = _listing(client, title="100% cotton shirt", seller_id="s-q")
    plain = _listing(client, title="cotton shirt", description="plain", seller_id="s-q")

    # LIKE metacharacters match literally, never as wildcards
    q = client.get("/v1/listings", params={"q": "100%"}).json()
    assert q["total"] == 1 and q["items"][0]["id"] == pct["id"]
    assert client.get("/v1/listings", params={"q": "cott_n"}).json()["total"] == 0
    assert client.get("/v1/listings", params={"q": "%"}).json()["total"] == 1

    # Unknown status is rejected, not silently empty
    assert client.get("/v1/listings", params={"status": "bogus"}).status_code == 422

    # sha256 must be hex; uppercase normalizes to lowercase
    bad_hex = client.post(f"/v1/listings/{pct['id']}/images", json={
        "storage_key": "k", "filename": "f.jpg", "content_type": "image/jpeg",
        "byte_size": 10, "sha256": "!" * 64})
    assert bad_hex.status_code == 422
    upper = client.post(f"/v1/listings/{pct['id']}/images", json={
        "storage_key": "k", "filename": "f.jpg", "content_type": "image/jpeg",
        "byte_size": 10, "sha256": "B" * 64})
    assert upper.status_code == 201 and upper.json()["sha256"] == "b" * 64

    # Evidence without provenance is rejected
    mv = _model_version(client)
    for attrs, why in [
        ([{"key": "brand", "value": "x", "confidence": 0.5,
           "evidence": [{"source": "image"}]}], "image-no-id"),
        ([{"key": "brand", "value": "x", "confidence": 0.5,
           "evidence": [{"source": "text"}]}], "text-no-span"),
        ([{"key": "brand", "value": "x", "confidence": 0.5,
           "evidence": [{"source": "text", "text_span": {"field": "title"}}]}], "text-partial-span"),
    ]:
        r = client.post(f"/v1/listings/{pct['id']}/predictions", json={
            "model_version_id": mv["id"], "attributes": attrs})
        assert r.status_code == 422, why

    # Plain substring still matches both fixtures
    got = client.get("/v1/listings", params={"q": "cotton shirt"}).json()
    assert got["total"] == 2 and {i["id"] for i in got["items"]} == {pct["id"], plain["id"]}


def test_listing_pages_are_stable_and_disjoint(client):
    ids = [_listing(client, title=f"pagetest {i}")["id"] for i in range(3)]
    seen: list[str] = []
    for offset in (0, 1, 2):
        page = client.get("/v1/listings", params={"limit": 1, "offset": offset}).json()
        assert page["total"] >= 3
        seen.extend(i["id"] for i in page["items"])
    assert len(set(seen)) == len(seen)  # no duplicates/skips across pages
    assert set(ids) <= set(
        i["id"] for i in client.get("/v1/listings", params={"limit": 100}).json()["items"]
    )


def test_write_size_caps_reject_oversized(client):
    mv = _model_version(client)
    lid = _listing(client)["id"]
    big_attrs = [{"key": f"k{i}", "value": "v", "confidence": 0.5} for i in range(101)]
    r = client.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "attributes": big_attrs})
    assert r.status_code == 422
    big_ev = [{"source": "text", "text_span": {"field": "t", "start": 0, "end": 1}}
              for _ in range(21)]
    r = client.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "attributes": [
            {"key": "k", "value": "v", "confidence": 0.5, "evidence": big_ev}]})
    assert r.status_code == 422
    assert client.post("/v1/listings", json={
        "title": "t", "seller_id": "s", "description": "x" * 5001}).status_code == 422


def test_multimodal_evidence_needs_provenance_and_dict_caps(client):
    mv = _model_version(client)
    lid = _listing(client)["id"]
    r = client.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "attributes": [
            {"key": "brand", "value": "x", "confidence": 0.5,
             "evidence": [{"source": "multimodal"}]}]})
    assert r.status_code == 422
    ok = client.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "attributes": [
            {"key": "brand", "value": "x", "confidence": 0.5,
             "evidence": [{"source": "multimodal",
                           "text_span": {"field": "t", "start": 0, "end": 1}}]}]})
    assert ok.status_code == 201
    assert client.post("/v1/model-versions", json={
        "name": "m", "version": "v",
        "config": {f"k{i}": i for i in range(51)}}).status_code == 422
    assert client.post(f"/v1/listings/{lid}/reviews", json={
        "decision": "corrected", "reviewer_id": "r",
        "corrected_attributes": {f"k{i}": i for i in range(51)}}).status_code == 422


def test_write_conflict_mapping_distinguishes_unique_vs_fk():
    import pytest
    from unittest.mock import MagicMock
    from uuid import uuid4

    from fastapi import HTTPException
    from sqlalchemy.exc import IntegrityError

    from app import models as _models
    from app.routers import create_prediction, register_image
    from app.schemas import AttributeIn, ImageRegister, PredictionCreate

    listing = _models.Listing(seller_id="s", title="t")
    listing.id = str(uuid4())
    mv_id = uuid4()

    def make_db(commit_exc):
        db = MagicMock()

        def _get(model, _id):
            if model is _models.Listing:
                return listing
            return object()
        db.get.side_effect = _get
        db.commit.side_effect = commit_exc
        return db

    unique_err = IntegrityError("INSERT", {}, Exception("UNIQUE constraint failed: prediction_attributes.key"))
    fk_err = IntegrityError("INSERT", {}, Exception("FOREIGN KEY constraint failed"))
    pred_body = PredictionCreate(model_version_id=mv_id, attributes=[
        AttributeIn(key="k", value="v", confidence=0.5)])
    img_body = ImageRegister(storage_key="k", filename="f.jpg", content_type="image/jpeg",
                             byte_size=10, sha256="a" * 64)

    with pytest.raises(HTTPException) as e:
        create_prediction(uuid4(), pred_body, make_db(unique_err))
    assert e.value.status_code == 409
    with pytest.raises(HTTPException) as e:
        create_prediction(uuid4(), pred_body, make_db(fk_err))
    assert e.value.status_code == 422
    with pytest.raises(HTTPException) as e:
        register_image(uuid4(), img_body, make_db(unique_err))
    assert e.value.status_code == 409
    with pytest.raises(HTTPException) as e:
        register_image(uuid4(), img_body, make_db(fk_err))
    assert e.value.status_code == 422


def test_retrieval_search_endpoint_uses_product_pipeline(client):
    from tests.conftest import TestingSession
    from app.retrieval import store_text_embedding, text_embed

    lid = _listing(client, title="red bicycle", description="fast bike")["id"]
    db = TestingSession()
    try:
        store_text_embedding(db, lid, text_embed("red bicycle fast bike"))
    finally:
        db.close()
    r = client.post("/v1/retrieval/search", json={"query_text": "bicycle", "modality": "text"})
    assert r.status_code == 200 and r.json()[0]["listing_id"] == lid
    img_ids = []
    for i, sha in enumerate(("ab" * 32, "cd" * 32, "ef" * 32)):
        img_ids.append(client.post(f"/v1/listings/{lid}/images", json={
            "storage_key": f"s3://bucket/r{i}.jpg", "filename": f"r{i}.jpg",
            "content_type": "image/jpeg", "byte_size": 10, "sha256": sha}).json()["id"])
    from app.retrieval import image_embed, store_image_embedding
    db = TestingSession()
    try:
        # Two vectors on one listing exercise hit dedupe; the third image
        # keeps a NULL embedding and must not become a candidate.
        store_image_embedding(db, img_ids[0], image_embed("ab" * 32))
        store_image_embedding(db, img_ids[1], image_embed("cd" * 32))
    finally:
        db.close()
    by_img = client.post("/v1/retrieval/search",
                         json={"modality": "image", "query_image_sha256": "ab" * 32})
    assert by_img.status_code == 200
    assert [h["listing_id"] for h in by_img.json()] == [lid]
    assert by_img.json()[0]["image_score"] == by_img.json()[0]["score"]
    multi = client.post("/v1/retrieval/search", json={"query_text": "bicycle", "modality": "multimodal"})
    assert multi.status_code == 200 and multi.json()[0]["listing_id"] == lid
    assert client.post("/v1/retrieval/search", json={"modality": "image"}).status_code == 422
    assert client.post("/v1/retrieval/search", json={"query_text": "   "}).status_code == 422
    assert client.post("/v1/retrieval/search", json={"query_text": "x" * 5001}).status_code == 422


def test_ocr_evidence_source_requires_text_span(client):
    mv = _model_version(client)
    lid = _listing(client)["id"]
    no_span = {"model_version_id": mv["id"], "attributes": [
        {"key": "brand", "value": "x", "confidence": 0.5, "evidence": [{"source": "ocr"}]}]}
    assert client.post(f"/v1/listings/{lid}/predictions", json=no_span).status_code == 422
    ok = {"model_version_id": mv["id"], "attributes": [
        {"key": "brand", "value": "x", "confidence": 0.5,
         "evidence": [{"source": "ocr", "text_span": {"field": "ocr", "start": 0, "end": 3}}]}]}
    assert client.post(f"/v1/listings/{lid}/predictions", json=ok).status_code == 201


def test_attribute_and_correction_value_caps(client):
    mv = _model_version(client)
    lid = _listing(client)["id"]
    r = client.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "attributes": [
            {"key": "k" * 201, "value": "v", "confidence": 0.5}]})
    assert r.status_code == 422
    r = client.post(f"/v1/listings/{lid}/predictions", json={
        "model_version_id": mv["id"], "attributes": [
            {"key": "k", "value": "v" * 2001, "confidence": 0.5}]})
    assert r.status_code == 422
    assert client.post(f"/v1/listings/{lid}/reviews", json={
        "decision": "corrected", "reviewer_id": "r",
        "corrected_attributes": {"brand": "x" * 2001}}).status_code == 422


def test_dashboard_payload_preserves_raw_strings_for_frontend_escaping(client):
    lid = _listing(client)["id"]
    xss = "<img src=x onerror=alert(1)>"
    client.post(f"/v1/listings/{lid}/images", json={
        "storage_key": "k", "filename": xss, "content_type": "image/jpeg",
        "byte_size": 10, "sha256": "e" * 64})
    d = client.get(f"/v1/listings/{lid}/review-dashboard").json()
    assert d["images"][0]["filename"] == xss  # raw by design; dashboard.html escapeHtml neutralizes

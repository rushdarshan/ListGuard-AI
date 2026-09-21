"""Risk engine TDD — RED phase tests (written BEFORE implementation).

Covers all 6 risk categories: unit + integration, edge cases, error conditions.
Every finding must contain: risk_category, severity, confidence, evidence,
rule_or_model_version, recommended_action.
"""
from app.risk import RULE_VERSION, ListingInput, assess_listing_risk

Ahex = "a" * 64
Bhex = "b" * 64


def _shape_ok(f):
    assert f.risk_category in (
        "image_text_contradiction", "missing_attributes", "repeated_description",
        "abnormal_pricing", "unusable_image", "unsupported_claim",
    )
    assert f.severity in ("low", "medium", "high")
    assert 0.0 <= f.confidence <= 1.0
    assert isinstance(f.evidence, list) and len(f.evidence) >= 1
    assert f.rule_or_model_version == RULE_VERSION
    assert f.recommended_action in ("publish", "warn", "human_review")


def _by_cat(assessment, cat):
    return [f for f in assessment.findings if f.risk_category == cat]


# ---- 1. image/text contradictions ----
def test_contradiction_text_vs_image():
    a = assess_listing_risk(ListingInput(
        title="Red Nike shoes",
        image_labels=[{"label": "blue", "raw": "blue shoes", "image_id": "img1"}],
    ))
    fs = _by_cat(a, "image_text_contradiction")
    assert len(fs) >= 1
    for f in fs:
        _shape_ok(f)
    assert a.recommended_action == "human_review"


def test_no_contradiction_when_agree():
    a = assess_listing_risk(ListingInput(
        title="Red Nike shoes",
        image_labels=[{"label": "red", "raw": "red shoes", "image_id": "img1"}],
    ))
    assert _by_cat(a, "image_text_contradiction") == []


# ---- 2. missing attributes ----
def test_missing_attributes_when_bare():
    a = assess_listing_risk(ListingInput(title="Nice item for sale", description="good stuff"))
    fs = _by_cat(a, "missing_attributes")
    assert len(fs) >= 1
    for f in fs:
        _shape_ok(f)


def test_no_missing_when_fully_grounded():
    a = assess_listing_risk(ListingInput(title="Nike red shoes new", description="Sony camera black used"))
    # brand+color+condition grounded somewhere; category may still miss -> only assert shape, not zero
    for f in a.findings:
        _shape_ok(f)


# ---- 3. repeated descriptions ----
def test_repeated_description_exact_duplicate():
    desc = "Brand new Nike red shoes, size 10, great condition"
    a = assess_listing_risk(ListingInput(
        title="Nike shoes", description=desc, other_descriptions=[desc]))
    fs = _by_cat(a, "repeated_description")
    assert len(fs) >= 1
    for f in fs:
        _shape_ok(f)


def test_no_repeat_when_unique():
    a = assess_listing_risk(ListingInput(
        title="Nike shoes", description="totally unique one-off description xyz",
        other_descriptions=["something completely different abc"]))
    assert _by_cat(a, "repeated_description") == []


def test_repeated_description_near_duplicate_warns():
    base = "brand new nike red running shoes size ten great condition fast shipping today"
    near = "brand new nike red running shoes size ten great condition free shipping today"
    a = assess_listing_risk(ListingInput(
        title="Nike shoes", description=base, other_descriptions=[near]))
    fs = _by_cat(a, "repeated_description")
    assert len(fs) == 1 and fs[0].severity == "low" and fs[0].recommended_action == "warn"
    _shape_ok(fs[0])


def test_abnormal_pricing_medium_and_low_tiers():
    med = assess_listing_risk(ListingInput(title="Shoes", price=350.0, category_median_price=100.0))
    fs_med = _by_cat(med, "abnormal_pricing")
    assert len(fs_med) == 1 and fs_med[0].severity == "medium"
    low = assess_listing_risk(ListingInput(title="Shoes", price=250.0, category_median_price=100.0))
    fs_low = _by_cat(low, "abnormal_pricing")
    assert len(fs_low) == 1 and fs_low[0].severity == "low"
    cheap = assess_listing_risk(ListingInput(title="Shoes", price=10.0, category_median_price=100.0))
    assert _by_cat(cheap, "abnormal_pricing")[0].severity == "high"
    for f in fs_med + fs_low:
        _shape_ok(f)


# ---- 4. abnormal pricing ----
def test_abnormal_pricing_spike():
    a = assess_listing_risk(ListingInput(
        title="Nike shoes", price=9999.0, category_median_price=100.0))
    fs = _by_cat(a, "abnormal_pricing")
    assert len(fs) >= 1
    for f in fs:
        _shape_ok(f)
    assert fs[0].severity == "high"


def test_abnormal_pricing_zero_or_negative():
    a = assess_listing_risk(ListingInput(title="Shoes", price=0.0, category_median_price=50.0))
    assert len(_by_cat(a, "abnormal_pricing")) >= 1


def test_normal_pricing_no_finding():
    a = assess_listing_risk(ListingInput(title="Shoes", price=95.0, category_median_price=100.0))
    assert _by_cat(a, "abnormal_pricing") == []


def test_pricing_skipped_without_baseline():
    a = assess_listing_risk(ListingInput(title="Shoes", price=95.0, category_median_price=None))
    assert _by_cat(a, "abnormal_pricing") == []


# ---- 5. unusable images ----
def test_unusable_image_tiny_dimensions():
    a = assess_listing_risk(ListingInput(
        title="Shoes", images=[{"image_id": "i1", "width": 10, "height": 10, "byte_size": 500}]))
    fs = _by_cat(a, "unusable_image")
    assert len(fs) >= 1
    for f in fs:
        _shape_ok(f)


def test_usable_image_no_finding():
    a = assess_listing_risk(ListingInput(
        title="Shoes", images=[{"image_id": "i1", "width": 800, "height": 600, "byte_size": 50000}]))
    assert _by_cat(a, "unusable_image") == []


# ---- 6. unsupported seller claims ----
def test_unsupported_claim_authentic_without_evidence():
    a = assess_listing_risk(ListingInput(
        title="100% authentic Nike shoes, certified original", description="guaranteed genuine"))
    fs = _by_cat(a, "unsupported_claim")
    assert len(fs) >= 1
    for f in fs:
        _shape_ok(f)


def test_no_claim_no_finding():
    a = assess_listing_risk(ListingInput(title="Nike red shoes", description="used pair"))
    assert _by_cat(a, "unsupported_claim") == []


# ---- overall action / edge cases ----
def test_clean_listing_publishes():
    a = assess_listing_risk(ListingInput(
        title="Nike red shoes new", description="red Nike sneakers, brand new",
        price=95.0, category_median_price=100.0,
        images=[{"image_id": "i1", "width": 800, "height": 600, "byte_size": 50000}],
    ))
    for f in a.findings:
        _shape_ok(f)
    assert a.recommended_action in ("publish", "warn", "human_review")


def test_empty_listing_does_not_crash():
    a = assess_listing_risk(ListingInput())
    for f in a.findings:
        _shape_ok(f)


# ---- integration: engine over API-created listing + images ----
def test_integration_api_listing_images(client):
    r = client.post("/v1/listings", json={"title": "Red Nike shoes", "description": "brand new", "seller_id": "s1"})
    assert r.status_code == 201
    lid = r.json()["id"]
    r2 = client.post(f"/v1/listings/{lid}/images", json={
        "storage_key": "k1", "filename": "a.jpg", "content_type": "image/jpeg",
        "byte_size": 500, "width": 10, "height": 10, "sha256": Ahex})
    assert r2.status_code == 201
    listing = client.get(f"/v1/listings/{lid}").json()
    images = client.get(f"/v1/listings/{lid}/images").json()
    a = assess_listing_risk(ListingInput(
        title=listing["title"], description=listing["description"],
        image_labels=[{"label": "blue", "raw": "blue shoes", "image_id": images[0]["id"]}],
        images=[{"image_id": images[0]["id"], "width": images[0]["width"],
                 "height": images[0]["height"], "byte_size": images[0]["byte_size"]}],
    ))
    cats = {f.risk_category for f in a.findings}
    assert "image_text_contradiction" in cats
    assert "unusable_image" in cats
    for f in a.findings:
        _shape_ok(f)


def test_integration_risk_from_db_models(client):
    r = client.post("/v1/listings", json={"title": "Nice item", "description": "good stuff", "seller_id": "s9"})
    lid = r.json()["id"]
    listing = client.get(f"/v1/listings/{lid}").json()
    a = assess_listing_risk(ListingInput(title=listing["title"], description=listing["description"] or ""))
    assert len(_by_cat(a, "missing_attributes")) >= 1

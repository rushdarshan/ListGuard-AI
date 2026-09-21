"""Grounded attribute extraction — RED tests (written BEFORE implementation).

Covers: brand/category/color/condition grounding, required prediction fields,
seller_text/ocr/image evidence, abstention, contradiction detection, raw refs,
adversarial anti-hallucination (brand/color/condition/size).
"""

from app.attributes import (
    MODEL_VERSION,
    SUPPORTED_KEYS,
    extract_attributes,
    to_prediction_payload,
)


def test_supported_keys():
    assert set(SUPPORTED_KEYS) == {"brand", "category", "color", "condition"}


def test_happy_path_seller_text_grounded():
    r = extract_attributes(title="Nike Air Red Shoes size 10", description="Brand new sneakers")
    assert r.attributes["brand"].value == "nike"
    assert r.attributes["brand"].abstained is False
    assert r.attributes["color"].value == "red"


def test_every_prediction_has_required_fields():
    r = extract_attributes(title="Sony black camera", description="used, great condition")
    for key in SUPPORTED_KEYS:
        p = r.attributes[key]
        assert p.key == key
        assert 0.0 <= p.confidence <= 1.0
        assert p.model_version == MODEL_VERSION
        assert isinstance(p.abstained, bool)
        assert isinstance(p.evidence, list)
        if p.abstained:
            assert p.value is None
        else:
            assert p.value is not None
            assert p.source in ("seller_text", "ocr", "image")
            assert len(p.evidence) >= 1
            # raw evidence preserved, value grounded in raw text
            raws = " ".join(e.raw for e in p.evidence).lower()
            assert p.value.lower() in raws


def test_ocr_evidence_grounds_brand():
    r = extract_attributes(title="Shoes", ocr_texts=["SONY box label"])
    assert r.attributes["brand"].value == "sony"
    assert any(e.source == "ocr" for e in r.attributes["brand"].evidence)


def test_image_evidence_grounds_color():
    r = extract_attributes(
        title="Jacket", image_labels=[{"label": "blue", "image_id": "img1", "raw": "blue jacket"}]
    )
    assert r.attributes["color"].value == "blue"
    assert any(e.source == "image" for e in r.attributes["color"].evidence)


def test_abstain_when_insufficient_evidence():
    r = extract_attributes(title="Nice item for sale", description="good stuff")
    for key in SUPPORTED_KEYS:
        assert r.attributes[key].abstained is True
        assert r.attributes[key].value is None
        assert r.attributes[key].confidence == 0.0


def test_empty_input_abstains_all():
    r = extract_attributes(title="", description="")
    for key in SUPPORTED_KEYS:
        assert r.attributes[key].abstained is True


def test_contradiction_text_vs_image_abstains_and_flags():
    r = extract_attributes(
        title="Red Nike shoes",
        image_labels=[{"label": "blue", "image_id": "img1", "raw": "blue shoes"}],
    )
    assert r.attributes["color"].abstained is True
    assert r.attributes["color"].value is None
    assert any(c["key"] == "color" for c in r.contradictions)


def test_condition_grounded_not_hallucinated():
    r = extract_attributes(description="used, some wear")
    assert r.attributes["condition"].value == "used"
    r2 = extract_attributes(description="great product must buy")
    assert r2.attributes["condition"].abstained is True


# ---- Adversarial: never present unsupported attributes as facts ----
def test_adversarial_hallucinated_brand():
    r = extract_attributes(title="Running shoes, lightweight", description="fast and comfy")
    assert r.attributes["brand"].abstained is True
    assert r.attributes["brand"].value is None


def test_adversarial_hallucinated_color():
    r = extract_attributes(title="Nike shoes", description="brand new pair")
    assert r.attributes["color"].abstained is True


def test_adversarial_hallucinated_condition():
    r = extract_attributes(title="Nike red shoes", description="nice pair")
    assert r.attributes["condition"].abstained is True


def test_adversarial_size_never_extracted():
    # size is NOT a supported key — must not appear even when present in text
    r = extract_attributes(title="Nike shoes size 10")
    assert "size" not in r.attributes
    assert set(r.attributes) == set(SUPPORTED_KEYS)


def test_to_prediction_payload_roundtrip():
    r = extract_attributes(title="Nike red shoes new")
    payload = to_prediction_payload(r)
    assert {a["key"] for a in payload} == set(SUPPORTED_KEYS)
    for a in payload:
        assert {"key", "value", "confidence", "abstained", "evidence"} <= set(a)
        if a["abstained"]:
            assert a["value"] is None
        else:
            assert a["value"] is not None and len(a["evidence"]) >= 1


def test_multi_source_agreement_boosts_confidence():
    r = extract_attributes(title="Nike shoes", ocr_texts=["nike box label"])
    p = r.attributes["brand"]
    assert p.value == "nike" and p.abstained is False
    assert p.confidence == 0.95
    assert {e.source for e in p.evidence} == {"seller_text", "ocr"}


def test_image_label_fallback_bbox_and_model_version():
    r = extract_attributes(
        title="Camera",
        image_labels=[{"label": "Sony", "image_id": "img9", "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4}}],
        model_version="custom-v2",
    )
    p = r.attributes["brand"]
    assert p.value == "sony" and p.model_version == "custom-v2"
    assert p.evidence[0].image_id == "img9"
    assert p.evidence[0].bounding_box == {"x": 1, "y": 2, "w": 3, "h": 4}
    payload = to_prediction_payload(r)
    assert payload[0]["model_version"] == "custom-v2"

"""Reviewer-dashboard assembly (GREEN — minimal stdlib-only aggregation).

One payload for the reviewer: listing + images + predictions (with
confidence, evidence, model version) + duplicate matches + price interval
+ risk findings + chronological decision history. Read-only assembly —
corrections are appended as review_events, predictions are never edited.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import math

from .risk import ListingInput, assess_listing_risk


def derive_price_interval(price_attributes: list[Any]) -> dict | None:
    """Min/max over numeric price-estimation values; None when none parse."""
    nums: list[float] = []
    for a in price_attributes:
        # Reviewer item 9: only explicit bound keys count. Any other numeric
        # attribute (similarity, confidence echoes) must not move the interval.
        key = a.get("key") if isinstance(a, dict) else getattr(a, "key", None)
        if key not in ("price_low", "price_high", "lower_bound", "upper_bound"):
            continue
        value = a.get("value") if isinstance(a, dict) else getattr(a, "value", None)
        try:
            f = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):  # NaN/inf would poison min/max
            nums.append(f)
    if not nums:
        return None
    return {"low": min(nums), "high": max(nums)}


def _attrs_of(pred: Any) -> list[Any]:
    return list(getattr(pred, "attributes", None) or [])


def _ev_dict(e: Any) -> dict:
    return {
        "id": e.id, "source": e.source, "image_id": e.image_id,
        "text_span": e.text_span, "bounding_box": e.bounding_box, "note": e.note,
    }


def assemble_dashboard(*, listing: Any, images: list[Any],
                       predictions: list[tuple[Any, Any]],
                       reviews: list[Any],
                       price: float | None = None,
                       category_median_price: float | None = None,
                       ocr_texts: list[str] | None = None,
                       image_labels: list[dict] | None = None,
                       other_descriptions: list[str] | None = None) -> dict:
    """Assemble the dashboard payload from already-loaded ORM rows.

    ``predictions`` is a list of ``(prediction, model_version)`` tuples with
    attributes + evidence eagerly loaded. Returns JSON-ready plain dicts.
    """
    pred_payloads: list[dict] = []
    duplicate_matches: list[dict] = []
    price_attrs: list[dict] = []
    for pred, mv in predictions:
        attrs = [
            {"id": a.id, "key": a.key, "value": a.value,
             "confidence": a.confidence, "abstained": a.abstained,
             "evidence": [_ev_dict(e) for e in (a.evidence or [])]}
            for a in _attrs_of(pred)
        ]
        pred_payloads.append({
            "id": pred.id, "prediction_type": pred.prediction_type,
            "latency_ms": pred.latency_ms, "created_at": pred.created_at,
            "model_version": {"id": mv.id, "name": mv.name, "version": mv.version},
            "attributes": attrs,
        })
        if pred.prediction_type == "duplicate_detection":
            for a in attrs:
                v = a["value"]
                if not isinstance(v, str) or not v:
                    continue  # abstained/non-string attrs are not matches
                duplicate_matches.append({
                    "prediction_id": pred.id,
                    "candidate_listing_id": v,
                    "score": a["confidence"],
                })
        elif pred.prediction_type == "price_estimation":
            price_attrs.extend(attrs)

    # Risk context: pass everything stored (category) plus caller-supplied
    # price/OCR/corpus context. The HTTP route currently has no stored price,
    # OCR texts, or description corpus, so those stay None/empty there until
    # the price model + OCR pipeline land — documented, not silently dropped.
    assessment = assess_listing_risk(ListingInput(
        title=listing.title or "", description=listing.description or "",
        price=price, category=listing.category,
        ocr_texts=list(ocr_texts or []), image_labels=list(image_labels or []),
        images=[{"image_id": i.id, "width": i.width,
                 "height": i.height, "byte_size": i.byte_size} for i in images],
        other_descriptions=list(other_descriptions or []),
        category_median_price=category_median_price,
    ))

    return {
        "listing": {
            "id": listing.id, "seller_id": listing.seller_id, "title": listing.title,
            "description": listing.description, "category": listing.category,
            "status": listing.status,
        },
        "images": [
            {"id": i.id, "listing_id": i.listing_id, "storage_key": i.storage_key,
             "filename": i.filename, "content_type": i.content_type,
             "byte_size": i.byte_size, "width": i.width, "height": i.height,
             "sha256": i.sha256, "created_at": i.created_at}
            for i in images
        ],
        "predictions": pred_payloads,
        "duplicate_matches": duplicate_matches,
        "price_interval": derive_price_interval(price_attrs),
        "risk_findings": [asdict(f) for f in assessment.findings],
        "decision_history": [
            {"id": r.id, "listing_id": r.listing_id, "prediction_id": r.prediction_id,
             "decision": r.decision, "corrected_attributes": r.corrected_attributes,
             "note": r.note, "reviewer_id": r.reviewer_id, "created_at": r.created_at}
            for r in reviews
        ],
    }

"""Grounded attribute extraction (GREEN — minimal stdlib-only implementation).

Every prediction carries value/confidence/source/evidence/model_version/abstained.
Evidence sources: seller_text | ocr | image. Deterministic closed-vocab matching;
a value is predicted ONLY if it appears as a whole word in raw evidence.
Conflicts across sources -> abstain + contradiction record. Learned models swap
in at the extract_attributes seam only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SUPPORTED_KEYS = ("brand", "category", "color", "condition")
MODEL_VERSION = "attributes-rule-v1"

EvidenceSource = str  # "seller_text" | "ocr" | "image"

SINGLE_SOURCE_CONF = 0.75
MULTI_SOURCE_CONF = 0.95

_VOCABS: dict[str, tuple[str, ...]] = {
    "brand": ("nike", "sony", "apple", "samsung", "adidas", "canon", "lego", "ikea"),
    "category": ("shoes", "sneakers", "camera", "jacket", "bicycle", "bike", "phone",
                 "laptop", "shirt", "dress", "watch", "bag", "toy", "sofa", "headphones"),
    "color": ("red", "blue", "black", "white", "green", "yellow", "pink", "purple",
              "orange", "gray", "grey", "brown", "beige", "navy"),
    "condition": ("new", "used", "refurbished"),
}


@dataclass
class Evidence:
    source: EvidenceSource
    raw: str
    field: str | None = None
    image_id: str | None = None
    span: tuple[int, int] | None = None
    bounding_box: dict | None = None


@dataclass
class AttributePrediction:
    key: str
    value: str | None
    confidence: float
    source: EvidenceSource | None
    evidence: list[Evidence] = field(default_factory=list)
    model_version: str = MODEL_VERSION
    abstained: bool = False


@dataclass
class ExtractionResult:
    attributes: dict[str, AttributePrediction] = field(default_factory=dict)
    contradictions: list[dict] = field(default_factory=list)


def _whole_word(haystack: str, needle: str) -> tuple[int, int] | None:
    m = re.search(r"\b" + re.escape(needle) + r"\b", haystack, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def _candidates_for(
    key: str,
    title: str,
    description: str,
    ocr_texts: list[str],
    image_labels: list[dict],
) -> list[tuple[str, Evidence]]:
    out: list[tuple[str, Evidence]] = []
    for value in _VOCABS[key]:
        for fname, text in (("title", title), ("description", description)):
            span = _whole_word(text, value) if text else None
            if span:
                out.append((value, Evidence(source="seller_text", raw=text,
                                           field=fname, span=span)))
        for ocr in ocr_texts:
            span = _whole_word(ocr, value) if ocr else None
            if span:
                out.append((value, Evidence(source="ocr", raw=ocr, span=span)))
        for item in image_labels:
            raw = str(item.get("raw") or item.get("label") or "")
            if raw and _whole_word(raw, value):
                out.append((value, Evidence(source="image", raw=raw,
                                           image_id=item.get("image_id"),
                                           bounding_box=item.get("bounding_box"))))
    return out


def extract_attributes(
    title: str | None = None,
    description: str | None = None,
    ocr_texts: list[str] | None = None,
    image_labels: list[dict] | None = None,
    model_version: str = MODEL_VERSION,
) -> ExtractionResult:
    title = title or ""
    description = description or ""
    ocr_texts = list(ocr_texts or [])
    image_labels = list(image_labels or [])
    result = ExtractionResult()
    for key in SUPPORTED_KEYS:
        cands = _candidates_for(key, title, description, ocr_texts, image_labels)
        by_value: dict[str, list[Evidence]] = {}
        for value, ev in cands:
            by_value.setdefault(value, []).append(ev)
        if not by_value:
            # ponytail: no grounding, no fact — abstain is the whole feature
            result.attributes[key] = AttributePrediction(
                key=key, value=None, confidence=0.0, source=None,
                evidence=[], model_version=model_version, abstained=True)
        elif len(by_value) == 1:
            value, evs = next(iter(by_value.items()))
            sources = {e.source for e in evs}
            result.attributes[key] = AttributePrediction(
                key=key, value=value,
                confidence=MULTI_SOURCE_CONF if len(sources) > 1 else SINGLE_SOURCE_CONF,
                source=next(e.source for e in evs), evidence=evs,
                model_version=model_version, abstained=False)
        else:
            evs = [e for ev_list in by_value.values() for e in ev_list]
            result.attributes[key] = AttributePrediction(
                key=key, value=None, confidence=0.0, source=None,
                evidence=evs, model_version=model_version, abstained=True)
            result.contradictions.append({
                "key": key,
                "values": sorted(by_value),
                "sources": sorted({e.source for e in evs}),
            })
    return result


def to_prediction_payload(result: ExtractionResult) -> list[dict]:
    payload = []
    for key in SUPPORTED_KEYS:
        p = result.attributes[key]
        payload.append({
            "key": p.key, "value": p.value, "confidence": p.confidence,
            "abstained": p.abstained, "source": p.source,
            "model_version": p.model_version,
            "evidence": [{"source": e.source, "raw": e.raw, "field": e.field,
                          "image_id": e.image_id, "span": list(e.span) if e.span else None,
                          "bounding_box": e.bounding_box} for e in p.evidence],
        })
    return payload

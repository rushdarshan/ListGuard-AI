"""Listing-quality and risk engine (GREEN — minimal stdlib-only rules).

Reuses app.attributes grounding for contradiction/missing detection; the rest
is deterministic thresholds. Every finding carries risk_category, severity,
confidence, evidence, rule_or_model_version, recommended_action.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .attributes import extract_attributes

RULE_VERSION = "risk-rule-v1"

RiskCategory = Literal[
    "image_text_contradiction",
    "missing_attributes",
    "repeated_description",
    "abnormal_pricing",
    "unusable_image",
    "unsupported_claim",
]
Severity = Literal["low", "medium", "high"]
RecommendedAction = Literal["publish", "warn", "human_review"]

MIN_IMAGE_DIM = 64
MIN_IMAGE_BYTES = 1024
REPEAT_JACCARD = 0.85

# ponytail: keyword list, not NLP — add learned claim verification when it misfires
_CLAIM_WORDS = ("authentic", "genuine", "certified", "guaranteed", "guarantee",
                "warranty", "official", "original")
_CLAIM_RE = re.compile(r"\b(" + "|".join(_CLAIM_WORDS) + r")\b", re.IGNORECASE)


@dataclass
class RiskEvidence:
    source: str
    note: str = ""
    image_id: str | None = None
    text_span: dict | None = None


@dataclass
class Finding:
    risk_category: RiskCategory
    severity: Severity
    confidence: float
    evidence: list[RiskEvidence]
    rule_or_model_version: str
    recommended_action: RecommendedAction
    message: str = ""


@dataclass
class ListingInput:
    title: str = ""
    description: str = ""
    price: float | None = None
    category: str | None = None
    ocr_texts: list[str] = field(default_factory=list)
    image_labels: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)  # metadata: {image_id?, width?, height?, byte_size?}
    other_descriptions: list[str] = field(default_factory=list)  # corpus for repeat check
    category_median_price: float | None = None


@dataclass
class RiskAssessment:
    findings: list[Finding] = field(default_factory=list)
    recommended_action: RecommendedAction = "publish"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _toks(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _finding(category: RiskCategory, severity: Severity, confidence: float,
             evidence: list[RiskEvidence], action: RecommendedAction, message: str) -> Finding:
    return Finding(risk_category=category, severity=severity, confidence=confidence,
                   evidence=evidence, rule_or_model_version=RULE_VERSION,
                   recommended_action=action, message=message)


def assess_listing_risk(listing: ListingInput) -> RiskAssessment:
    findings: list[Finding] = []
    title, desc = listing.title or "", listing.description or ""

    # 1+2. contradictions + missing attributes via grounded extraction
    # ponytail: shoes/sneakers synonyms count as conflict; alias map if it matters
    extracted = extract_attributes(title=title, description=desc,
                                   ocr_texts=listing.ocr_texts,
                                   image_labels=listing.image_labels)
    contradicted = set()
    for c in extracted.contradictions:
        contradicted.add(c["key"])
        findings.append(_finding(
            "image_text_contradiction", "high", 0.9,
            [RiskEvidence(source=s, note=f"{c['key']}: conflicting values {c['values']}")
             for s in c["sources"]] or [RiskEvidence(source="multimodal", note=str(c))],
            "human_review", f"conflicting {c['key']}: {', '.join(c['values'])}"))
    missing = [k for k, p in extracted.attributes.items() if p.abstained and k not in contradicted]
    if missing:
        findings.append(_finding(
            "missing_attributes", "medium", 0.8,
            [RiskEvidence(source="seller_text", note=f"no grounded value for '{k}'") for k in missing],
            "warn", f"missing attributes: {', '.join(sorted(missing))}"))

    # 3. repeated descriptions
    needle = _norm(desc)
    if needle:
        for other in listing.other_descriptions:
            if _norm(other) == needle:
                findings.append(_finding(
                    "repeated_description", "medium", 0.95,
                    [RiskEvidence(source="description_corpus", note="exact normalized duplicate")],
                    "human_review", "description duplicates another listing"))
                break
        else:
            nt = _toks(desc)
            for other in listing.other_descriptions:
                ot = _toks(other)
                if nt and ot and len(nt & ot) / len(nt | ot) >= REPEAT_JACCARD:
                    findings.append(_finding(
                        "repeated_description", "low", 0.7,
                        [RiskEvidence(source="description_corpus", note="near-duplicate (jaccard>=0.85)")],
                        "warn", "description near-duplicates another listing"))
                    break

    # 4. abnormal pricing
    price, median = listing.price, listing.category_median_price
    if price is not None and median:
        if price <= 0:
            findings.append(_finding(
                "abnormal_pricing", "high", 0.95,
                [RiskEvidence(source="price", note=f"non-positive price {price}")],
                "human_review", f"invalid price {price}"))
        else:
            ratio = price / median
            if ratio >= 5 or ratio <= 0.2:
                findings.append(_finding(
                    "abnormal_pricing", "high", 0.85,
                    [RiskEvidence(source="price", note=f"price {price} vs median {median} (x{ratio:.2f})")],
                    "human_review", f"price x{ratio:.1f} vs category median"))
            elif ratio >= 3 or ratio <= 1 / 3:
                findings.append(_finding(
                    "abnormal_pricing", "medium", 0.75,
                    [RiskEvidence(source="price", note=f"price {price} vs median {median} (x{ratio:.2f})")],
                    "warn", f"price x{ratio:.1f} vs category median"))
            elif ratio >= 2 or ratio <= 0.5:
                findings.append(_finding(
                    "abnormal_pricing", "low", 0.65,
                    [RiskEvidence(source="price", note=f"price {price} vs median {median} (x{ratio:.2f})")],
                    "warn", f"price x{ratio:.1f} vs category median"))

    # 5. unusable images
    bad = 0
    for img in listing.images:
        reasons = []
        w, h, n = img.get("width"), img.get("height"), img.get("byte_size")
        if isinstance(w, (int, float)) and w < MIN_IMAGE_DIM:
            reasons.append(f"width {w}<{MIN_IMAGE_DIM}")
        if isinstance(h, (int, float)) and h < MIN_IMAGE_DIM:
            reasons.append(f"height {h}<{MIN_IMAGE_DIM}")
        if isinstance(n, (int, float)) and n < MIN_IMAGE_BYTES:
            reasons.append(f"byte_size {n}<{MIN_IMAGE_BYTES}")
        if reasons:
            bad += 1
            findings.append(_finding(
                "unusable_image", "medium", 0.9,
                [RiskEvidence(source="image_metadata", note="; ".join(reasons),
                              image_id=img.get("image_id"))],
                "warn", f"unusable image {img.get('image_id')}: {'; '.join(reasons)}"))
    if listing.images and bad == len(listing.images):  # nothing usable left
        for f in findings:
            if f.risk_category == "unusable_image":
                f.severity, f.recommended_action = "high", "human_review"

    # 6. unsupported seller claims
    for m in {m.group(0).lower() for m in _CLAIM_RE.finditer(f"{title} {desc}")}:
        findings.append(_finding(
            "unsupported_claim", "high", 0.75,
            [RiskEvidence(source="seller_text", note=f"claim '{m}' has no supporting evidence")],
            "human_review", f"unsupported claim: '{m}' requires proof"))

    action: RecommendedAction = "publish"
    if any(f.recommended_action == "human_review" for f in findings):
        action = "human_review"
    elif any(f.recommended_action == "warn" for f in findings):
        action = "warn"
    return RiskAssessment(findings=findings, recommended_action=action)

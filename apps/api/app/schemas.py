from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

PredictionType = Literal["attribute_extraction", "duplicate_detection", "price_estimation", "quality_check"]
EvidenceSource = Literal["text", "image", "multimodal", "ocr"]


class RetrievalSearchIn(BaseModel):
    query_text: str | None = Field(default=None, max_length=5000)
    query_image_sha256: str | None = Field(default=None, pattern="^[0-9a-fA-F]{64}$")
    modality: Literal["text", "image", "multimodal"] = "multimodal"
    top_k: int = Field(default=10, ge=1, le=100)


class RetrievalHit(BaseModel):
    listing_id: UUID
    score: float
    text_score: float | None = None
    image_score: float | None = None
ReviewDecision = Literal["accepted", "rejected", "corrected"]
ListingStatus = Literal["draft", "pending_review", "approved", "rejected"]


# ---- Listings ----
class ListingCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    category: str | None = Field(default=None, max_length=120)
    seller_id: str = Field(min_length=1, max_length=200)


class ListingUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    category: str | None = Field(default=None, max_length=120)
    status: ListingStatus | None = None


class ListingOut(BaseModel):
    id: UUID
    seller_id: str
    title: str
    description: str | None
    category: str | None
    status: str
    created_at: datetime
    updated_at: datetime


class ListingPage(BaseModel):
    items: list[ListingOut]
    total: int
    limit: int
    offset: int


# ---- Images (metadata registration) ----
class ImageRegister(BaseModel):
    storage_key: str = Field(min_length=1, max_length=512, description="Key of an object ALREADY in object/local storage")
    filename: str = Field(min_length=1, max_length=512)
    content_type: Literal["image/jpeg", "image/png", "image/webp"]
    byte_size: int = Field(gt=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    sha256: str = Field(pattern="^[0-9a-fA-F]{64}$")


class ImageOut(ImageRegister):
    id: UUID
    listing_id: UUID
    created_at: datetime


# ---- Model versions ----
class ModelVersionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=200)
    config: dict = Field(default_factory=dict)

    @field_validator("config")
    @classmethod
    def _cap_config(cls, v: dict) -> dict:
        if len(v) > 50:
            raise ValueError("config must have at most 50 keys")
        return v


class ModelVersionOut(ModelVersionCreate):
    id: UUID
    created_at: datetime


# ---- Predictions ----
class EvidenceIn(BaseModel):
    source: EvidenceSource
    image_id: UUID | None = None
    text_span: dict | None = None  # {field, start, end}
    bounding_box: dict | None = None  # {x, y, w, h}
    note: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _require_provenance(self):
        if self.source == "image" and self.image_id is None:
            raise ValueError("source 'image' requires image_id")
        if self.source == "text" and not (
            isinstance(self.text_span, dict) and {"field", "start", "end"} <= set(self.text_span)
        ):
            raise ValueError("source 'text' requires text_span with field/start/end")
        if self.source == "multimodal" and self.image_id is None and not self.text_span:
            raise ValueError("source 'multimodal' requires image_id and/or text_span")
        if self.source == "ocr" and not (
            isinstance(self.text_span, dict) and {"field", "start", "end"} <= set(self.text_span)
        ):
            raise ValueError("source 'ocr' requires text_span with field/start/end")
        return self


class AttributeIn(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    value: str | None = Field(default=None, max_length=2000)
    confidence: float = Field(ge=0, le=1)
    abstained: bool = False
    evidence: list[EvidenceIn] = Field(default_factory=list, max_length=20)


class PredictionCreate(BaseModel):
    model_version_id: UUID
    prediction_type: PredictionType = "attribute_extraction"
    latency_ms: int | None = Field(default=None, ge=0)
    attributes: list[AttributeIn] = Field(min_length=1, max_length=100)


class EvidenceOut(EvidenceIn):
    id: UUID


class AttributeOut(BaseModel):
    id: UUID
    key: str
    value: str | None
    confidence: float
    abstained: bool
    evidence: list[EvidenceOut]


class PredictionOut(BaseModel):
    id: UUID
    listing_id: UUID
    model_version_id: UUID
    prediction_type: str
    latency_ms: int | None
    created_at: datetime
    attributes: list[AttributeOut]


# ---- Reviews ----
class ReviewCreate(BaseModel):
    decision: ReviewDecision
    prediction_id: UUID | None = None
    corrected_attributes: dict | None = None
    note: str | None = Field(default=None, max_length=2000)
    reviewer_id: str = Field(min_length=1, max_length=200)

    @field_validator("corrected_attributes")
    @classmethod
    def _cap_corrected(cls, v: dict | None) -> dict | None:
        if v is not None and len(v) > 50:
            raise ValueError("corrected_attributes must have at most 50 keys")
        if v is not None:
            for v2 in v.values():
                if isinstance(v2, str) and len(v2) > 2000:
                    raise ValueError("corrected_attributes values must be at most 2000 chars")
        return v


class ReviewOut(ReviewCreate):
    id: UUID
    listing_id: UUID
    created_at: datetime

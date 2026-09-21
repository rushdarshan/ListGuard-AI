from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from . import models, schemas
from .core.db import get_db
from .retrieval import (
    image_embed,
    image_similarity_search,
    multimodal_retrieve,
    text_embed,
    text_vector_search,
)

router = APIRouter(prefix="/v1")

ALLOWED_STATUSES = ("draft", "pending_review", "approved", "rejected")


def _escape_like(raw: str) -> str:
    return raw.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _listing_or_404(db: Session, listing_id: UUID) -> models.Listing:
    listing = db.get(models.Listing, str(listing_id))
    if listing is None or listing.deleted_at is not None:
        raise HTTPException(status_code=404, detail="listing not found")
    return listing


def _to_listing_out(row: models.Listing) -> schemas.ListingOut:
    return schemas.ListingOut.model_validate({
        "id": row.id, "seller_id": row.seller_id, "title": row.title, "description": row.description,
        "category": row.category, "status": row.status, "created_at": row.created_at, "updated_at": row.updated_at,
    })


def _to_image_out(r: models.ListingImage) -> schemas.ImageOut:
    return schemas.ImageOut.model_validate({
        "id": r.id, "listing_id": r.listing_id, "storage_key": r.storage_key, "filename": r.filename,
        "content_type": r.content_type, "byte_size": r.byte_size, "width": r.width, "height": r.height,
        "sha256": r.sha256, "created_at": r.created_at,
    })


def _to_model_version_out(r: models.ModelVersion) -> schemas.ModelVersionOut:
    return schemas.ModelVersionOut.model_validate({
        "id": r.id, "name": r.name, "version": r.version, "config": r.config, "created_at": r.created_at,
    })


def _to_review_out(r: models.ReviewEvent) -> schemas.ReviewOut:
    return schemas.ReviewOut.model_validate({
        "id": r.id, "listing_id": r.listing_id, "prediction_id": r.prediction_id, "decision": r.decision,
        "corrected_attributes": r.corrected_attributes, "note": r.note,
        "reviewer_id": r.reviewer_id, "created_at": r.created_at,
    })


# ---- Listings ----
@router.post("/listings", response_model=schemas.ListingOut, status_code=201)
def create_listing(body: schemas.ListingCreate, db: Session = Depends(get_db)):
    row = models.Listing(seller_id=body.seller_id, title=body.title, description=body.description, category=body.category)
    db.add(row)
    db.commit()
    db.refresh(row)
    return _to_listing_out(row)


@router.get("/listings", response_model=schemas.ListingPage)
def list_listings(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: str | None = None,
    seller_id: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db),
):
    stmt = select(models.Listing).where(models.Listing.deleted_at.is_(None))
    count_stmt = select(func.count()).select_from(models.Listing).where(models.Listing.deleted_at.is_(None))
    if status is not None:
        if status not in ALLOWED_STATUSES:
            raise HTTPException(status_code=422, detail=f"unknown status '{status}'")
        stmt = stmt.where(models.Listing.status == status)
        count_stmt = count_stmt.where(models.Listing.status == status)
    if seller_id:
        stmt = stmt.where(models.Listing.seller_id == seller_id)
        count_stmt = count_stmt.where(models.Listing.seller_id == seller_id)
    if q:  # portable literal-substring pre-filter; LIKE metacharacters escaped
        like = f"%{_escape_like(q)}%"
        stmt = stmt.where(or_(
            models.Listing.title.ilike(like, escape="\\"),
            models.Listing.description.ilike(like, escape="\\"),
        ))
        count_stmt = count_stmt.where(or_(
            models.Listing.title.ilike(like, escape="\\"),
            models.Listing.description.ilike(like, escape="\\"),
        ))
    total = db.scalar(count_stmt) or 0
    rows = db.scalars(stmt.order_by(models.Listing.created_at.desc(), models.Listing.id.desc()).limit(limit).offset(offset)).all()
    return schemas.ListingPage(items=[_to_listing_out(r) for r in rows], total=total, limit=limit, offset=offset)


@router.get("/listings/{listing_id}", response_model=schemas.ListingOut)
def get_listing(listing_id: UUID, db: Session = Depends(get_db)):
    return _to_listing_out(_listing_or_404(db, listing_id))


@router.patch("/listings/{listing_id}", response_model=schemas.ListingOut)
def update_listing(listing_id: UUID, body: schemas.ListingUpdate, db: Session = Depends(get_db)):
    row = _listing_or_404(db, listing_id)
    for field in ("title", "description", "category", "status"):
        value = getattr(body, field)
        if value is not None:
            setattr(row, field, value)
    db.commit()
    db.refresh(row)
    return _to_listing_out(row)


@router.delete("/listings/{listing_id}", status_code=204)
def delete_listing(listing_id: UUID, db: Session = Depends(get_db)):
    from datetime import datetime, timezone

    row = _listing_or_404(db, listing_id)
    row.deleted_at = datetime.now(timezone.utc)
    db.commit()
    return Response(status_code=204)


# ---- Images: metadata REGISTRATION (no bytes accepted here) ----
@router.post("/retrieval/search", response_model=list[schemas.RetrievalHit])
def retrieval_search(body: schemas.RetrievalSearchIn, db: Session = Depends(get_db)):
    """Execute the retrieval pipeline over stored fixture embeddings (reviewer item 3).

    Vectors are deterministic fixtures until a learned model lands; the route,
    ranking, evidence, and latency shape are the production contract.
    """
    if body.modality == "image":
        if not body.query_image_sha256:
            raise HTTPException(status_code=422, detail="modality 'image' requires query_image_sha256")
        # Unreachable-guard removed: the schema pattern rejects non-hex shas
        # with 422 before this line, so image_embed cannot raise here.
        qvec = image_embed(body.query_image_sha256)
        img_rows = (
            db.query(models.ListingImage)
            .join(models.Listing, models.ListingImage.listing_id == models.Listing.id)
            .filter(models.Listing.deleted_at.is_(None), models.ListingImage.image_embedding.is_not(None))
            .all()
        )
        cands = [{"id": r.listing_id, "vector": list(r.image_embedding or [])}
                 for r in img_rows if r.image_embedding]
        seen: set[str] = set()
        hits: list[schemas.RetrievalHit] = []
        for lid, s in image_similarity_search(qvec, cands, top_k=body.top_k):
            if lid in seen:
                continue
            seen.add(lid)
            hits.append(schemas.RetrievalHit.model_validate(
                {"listing_id": lid, "score": s, "text_score": None, "image_score": s}))
        return hits
    if not (body.query_text or "").strip():
        raise HTTPException(status_code=422, detail="modality 'text'/'multimodal' requires query_text")
    if body.modality == "text":
        # Same reasoning: blank text is rejected above, text_embed cannot raise.
        qvec = text_embed(body.query_text or "")
        ranked = text_vector_search(db, qvec, top_k=body.top_k)
        return [schemas.RetrievalHit.model_validate(
            {"listing_id": lid, "score": s, "text_score": s, "image_score": None})
                for lid, s in ranked]
    sha_by_listing: dict[str, str] = {}
    for im in db.query(models.ListingImage).all():
        sha_by_listing.setdefault(im.listing_id, im.sha256)
    docs = [
        {"id": r.id, "title": r.title or "", "description": r.description or "",
         "image_sha256": sha_by_listing.get(r.id)}
        for r in db.query(models.Listing).filter(models.Listing.deleted_at.is_(None)).all()
    ]
    res = multimodal_retrieve(body.query_text or "", image_sha256=body.query_image_sha256,
                              docs=docs, top_k=body.top_k)
    return [schemas.RetrievalHit.model_validate({"listing_id": h.id, "score": h.score})
            for h in res.results]


@router.post("/listings/{listing_id}/images", response_model=schemas.ImageOut, status_code=201)
def register_image(listing_id: UUID, body: schemas.ImageRegister, db: Session = Depends(get_db)):
    _listing_or_404(db, listing_id)
    row = models.ListingImage(
        listing_id=str(listing_id), storage_key=body.storage_key, filename=body.filename,
        content_type=body.content_type, byte_size=body.byte_size, width=body.width,
        height=body.height, sha256=body.sha256.lower(),
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        if "unique" in str(getattr(exc, "orig", exc)).lower():
            raise HTTPException(status_code=409, detail="duplicate image sha256 for this listing")
        raise HTTPException(status_code=422, detail="write conflict, please retry")
    db.refresh(row)
    return _to_image_out(row)


@router.get("/listings/{listing_id}/images", response_model=list[schemas.ImageOut])
def list_images(listing_id: UUID, db: Session = Depends(get_db)):
    _listing_or_404(db, listing_id)
    rows = db.scalars(select(models.ListingImage).where(models.ListingImage.listing_id == str(listing_id))).all()
    return [_to_image_out(r) for r in rows]


# ---- Model versions ----
@router.post("/model-versions", response_model=schemas.ModelVersionOut, status_code=201)
def create_model_version(body: schemas.ModelVersionCreate, db: Session = Depends(get_db)):
    row = models.ModelVersion(name=body.name, version=body.version, config=body.config)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="model name+version already exists")
    db.refresh(row)
    return _to_model_version_out(row)


@router.get("/model-versions", response_model=list[schemas.ModelVersionOut])
def list_model_versions(db: Session = Depends(get_db)):
    rows = db.scalars(select(models.ModelVersion).order_by(models.ModelVersion.created_at.desc())).all()
    return [_to_model_version_out(r) for r in rows]


# ---- Predictions ----
def _prediction_out(pred: models.Prediction) -> schemas.PredictionOut:
    attrs = []
    for a in pred.attributes:
        ev = [
            schemas.EvidenceOut.model_validate({
                "id": e.id, "source": e.source, "image_id": e.image_id,
                "text_span": e.text_span, "bounding_box": e.bounding_box, "note": e.note,
            })
            for e in a.evidence
        ]
        attrs.append(schemas.AttributeOut.model_validate({
            "id": a.id, "key": a.key, "value": a.value, "confidence": a.confidence,
            "abstained": a.abstained, "evidence": [e.model_dump() for e in ev],
        }))
    return schemas.PredictionOut.model_validate({
        "id": pred.id, "listing_id": pred.listing_id, "model_version_id": pred.model_version_id,
        "prediction_type": pred.prediction_type, "latency_ms": pred.latency_ms,
        "created_at": pred.created_at, "attributes": [a.model_dump() for a in attrs],
    })


@router.post("/listings/{listing_id}/predictions", response_model=schemas.PredictionOut, status_code=201)
def create_prediction(listing_id: UUID, body: schemas.PredictionCreate, db: Session = Depends(get_db)):
    _listing_or_404(db, listing_id)
    if db.get(models.ModelVersion, str(body.model_version_id)) is None:
        raise HTTPException(status_code=404, detail="model version not found")
    # Abstention invariant at API layer (DB CHECK is the backstop)
    for attr in body.attributes:
        if attr.abstained and attr.value is not None:
            raise HTTPException(status_code=422, detail=f"attribute '{attr.key}': abstained requires value=null")
        if not attr.abstained and attr.value is None:
            raise HTTPException(status_code=422, detail=f"attribute '{attr.key}': non-abstained requires a value")
    # All validation happens before any write: no partial flushes on 4xx paths.
    for attr in body.attributes:
        for e in attr.evidence:
            if e.image_id is not None:
                img = db.get(models.ListingImage, str(e.image_id))
                if img is None or img.listing_id != str(listing_id):
                    raise HTTPException(status_code=422, detail="evidence image_id must belong to this listing")
    pred = models.Prediction(
        listing_id=str(listing_id), model_version_id=str(body.model_version_id),
        prediction_type=body.prediction_type, latency_ms=body.latency_ms,
    )
    db.add(pred)
    db.flush()
    try:
        for attr in body.attributes:
            a = models.PredictionAttribute(
                prediction_id=pred.id, key=attr.key, value=attr.value,
                confidence=attr.confidence, abstained=attr.abstained,
            )
            db.add(a)
            db.flush()
            for e in attr.evidence:
                db.add(models.PredictionEvidence(
                    prediction_attribute_id=a.id, source=e.source, image_id=str(e.image_id) if e.image_id else None,
                    text_span=e.text_span, bounding_box=e.bounding_box, note=e.note,
                ))
        db.commit()
    except IntegrityError as exc:
        # Duplicate-key is the expected cause; anything else is a race
        # (e.g. referenced image deleted mid-flight) — report honestly.
        db.rollback()
        if "unique" in str(getattr(exc, "orig", exc)).lower():
            raise HTTPException(status_code=409, detail="duplicate attribute key in prediction")
        raise HTTPException(status_code=422, detail="write conflict, please retry")
    pred = db.scalars(
        select(models.Prediction)
        .where(models.Prediction.id == pred.id)
        .options(selectinload(models.Prediction.attributes).selectinload(models.PredictionAttribute.evidence))
    ).one()
    return _prediction_out(pred)


@router.get("/listings/{listing_id}/predictions", response_model=list[schemas.PredictionOut])
def list_predictions(listing_id: UUID, db: Session = Depends(get_db)):
    _listing_or_404(db, listing_id)
    preds = db.scalars(
        select(models.Prediction)
        .where(models.Prediction.listing_id == str(listing_id))
        .order_by(models.Prediction.created_at.desc(), models.Prediction.id.desc())
        .options(selectinload(models.Prediction.attributes).selectinload(models.PredictionAttribute.evidence))
    ).all()
    return [_prediction_out(p) for p in preds]


# ---- Reviews (append-only) ----
@router.post("/listings/{listing_id}/reviews", response_model=schemas.ReviewOut, status_code=201)
def create_review(listing_id: UUID, body: schemas.ReviewCreate, db: Session = Depends(get_db)):
    _listing_or_404(db, listing_id)
    if body.decision == "corrected" and not body.corrected_attributes:
        raise HTTPException(status_code=422, detail="corrected decision requires corrected_attributes")
    if body.prediction_id is not None:
        pred = db.get(models.Prediction, str(body.prediction_id))
        if pred is None or pred.listing_id != str(listing_id):
            raise HTTPException(status_code=422, detail="prediction_id must belong to this listing")
    row = models.ReviewEvent(
        listing_id=str(listing_id),
        prediction_id=str(body.prediction_id) if body.prediction_id else None,
        reviewer_id=body.reviewer_id, decision=body.decision,
        corrected_attributes=body.corrected_attributes, note=body.note,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _to_review_out(row)


@router.get("/listings/{listing_id}/reviews", response_model=list[schemas.ReviewOut])
def list_reviews(listing_id: UUID, db: Session = Depends(get_db)):
    _listing_or_404(db, listing_id)
    rows = db.scalars(
        select(models.ReviewEvent)
        .where(models.ReviewEvent.listing_id == str(listing_id))
        .order_by(models.ReviewEvent.created_at.asc(), models.ReviewEvent.id.asc())
    ).all()
    return [_to_review_out(r) for r in rows]


# ---- Reviewer dashboard (read-only aggregate; corrections stay append-only) ----
@router.get("/listings/{listing_id}/review-dashboard")
def review_dashboard(listing_id: UUID, db: Session = Depends(get_db)):
    from .dashboard import assemble_dashboard

    listing = _listing_or_404(db, listing_id)
    images = db.scalars(
        select(models.ListingImage).where(models.ListingImage.listing_id == str(listing_id))
    ).all()
    preds = db.scalars(
        select(models.Prediction)
        .where(models.Prediction.listing_id == str(listing_id))
        .order_by(models.Prediction.created_at.desc(), models.Prediction.id.desc())
        .options(selectinload(models.Prediction.attributes).selectinload(models.PredictionAttribute.evidence))
    ).all()
    mv_ids = {p.model_version_id for p in preds}
    mvs = {r.id: r for r in db.scalars(
        select(models.ModelVersion).where(models.ModelVersion.id.in_(mv_ids))).all()} if mv_ids else {}
    reviews = db.scalars(
        select(models.ReviewEvent)
        .where(models.ReviewEvent.listing_id == str(listing_id))
        .order_by(models.ReviewEvent.created_at.asc(), models.ReviewEvent.id.asc())
    ).all()
    return assemble_dashboard(
        listing=listing, images=list(images),
        predictions=[(p, mvs[p.model_version_id]) for p in preds], reviews=list(reviews))

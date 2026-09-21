"""Duplicate-detection evaluation pipeline (eval-only, no inference weights).

Compares text-only / image-only / multimodal retrieval over a versioned,
labeled dataset and reports Recall@K, MRR, mAP plus an ablation report.
All numbers are MEASURED from the retriever under test — never invented.
Retrieval itself reuses app.retrieval fixture embeddings; learned weights
swap in at the text_embed/image_embed seam only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Modality = Literal["text", "image", "multimodal"]


@dataclass
class EvalQuery:
    query_id: str
    text: str
    relevant_ids: list[str] = field(default_factory=list)
    image_sha256: str | None = None


@dataclass
class EvalDataset:
    version: str
    queries: list[EvalQuery] = field(default_factory=list)


@dataclass
class EvalMetrics:
    modality: Modality
    dataset_version: str
    model_version: str
    k: int
    n_queries: int
    recall_at_k: float
    mrr: float
    map: float


@dataclass
class AblationReport:
    dataset_version: str
    model_version: str
    k: int
    per_modality: dict[str, EvalMetrics] = field(default_factory=dict)


def _check_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive int")


def recall_at_k(relevant: list[str], ranked: list[str], k: int) -> float:
    _check_k(k)
    if not relevant:
        return 0.0
    return len(set(relevant) & set(ranked[:k])) / len(set(relevant))


def reciprocal_rank(relevant: list[str], ranked: list[str]) -> float:
    rel = set(relevant)
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0


def average_precision(relevant: list[str], ranked: list[str]) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def evaluate_rankings(rankings: dict[str, list[str]], dataset: EvalDataset, k: int) -> tuple[float, float, float]:
    _check_k(k)
    if not dataset.queries:
        raise ValueError("dataset has no queries")
    rs = [recall_at_k(q.relevant_ids, rankings.get(q.query_id, []), k) for q in dataset.queries]
    rr = [reciprocal_rank(q.relevant_ids, rankings.get(q.query_id, [])) for q in dataset.queries]
    ap = [average_precision(q.relevant_ids, rankings.get(q.query_id, [])) for q in dataset.queries]
    n = len(dataset.queries)
    return sum(rs) / n, sum(rr) / n, sum(ap) / n


def retrieve_ranked_ids(
    query: EvalQuery,
    docs: list[dict[str, Any]],
    modality: Modality,
    top_k: int,
    text_weight: float = 0.7,
) -> list[str]:
    from . import retrieval as _r  # ponytail: lazy import, reuse fixture embeddings

    if modality == "text":
        return [d["id"] for d in _r.full_text_search(query.text, docs, top_k=top_k)]
    if modality == "image":
        if not query.image_sha256:
            raise ValueError("image modality needs query.image_sha256")
        cands = [{"id": d["id"], "vector": _r.image_embed(d["image_sha256"])} for d in docs if d.get("image_sha256")]
        return [doc_id for doc_id, _ in _r.image_similarity_search(_r.image_embed(query.image_sha256), cands, top_k=top_k)]
    if modality == "multimodal":
        res = _r.multimodal_retrieve(query.text, docs, image_sha256=query.image_sha256, top_k=top_k, text_weight=text_weight)
        return [item.id for item in res.results]
    raise ValueError(f"unknown modality: {modality!r}")


def evaluate_modality(
    dataset: EvalDataset,
    docs: list[dict[str, Any]],
    modality: Modality,
    k: int = 10,
    model_version: str = "fixture-v1",
    text_weight: float = 0.7,
    top_k: int | None = None,
) -> EvalMetrics:
    _check_k(k)
    if not dataset.queries:
        raise ValueError("dataset has no queries")
    depth = top_k if top_k is not None else k
    rankings = {q.query_id: retrieve_ranked_ids(q, docs, modality, top_k=depth, text_weight=text_weight) for q in dataset.queries}
    r, mrr, m = evaluate_rankings(rankings, dataset, k)
    return EvalMetrics(modality=modality, dataset_version=dataset.version, model_version=model_version, k=k, n_queries=len(dataset.queries), recall_at_k=r, mrr=mrr, map=m)


def ablation_report(
    dataset: EvalDataset,
    docs: list[dict[str, Any]],
    k: int = 10,
    model_version: str = "fixture-v1",
) -> AblationReport:
    modalities: tuple[Modality, ...] = ("text", "image", "multimodal")
    per_modality: dict[str, EvalMetrics] = {}
    for m in modalities:
        try:
            per_modality[m] = evaluate_modality(dataset, docs, m, k=k, model_version=model_version)
        except ValueError:
            # e.g. image modality on a text-only dataset: record unevaluated
            # (n_queries=0) rather than aborting the whole report or
            # fabricating metric values.
            per_modality[m] = EvalMetrics(modality=m, dataset_version=dataset.version,
                                          model_version=model_version, k=k, n_queries=0,
                                          recall_at_k=0.0, mrr=0.0, map=0.0)
    return AblationReport(
        dataset_version=dataset.version,
        model_version=model_version,
        k=k,
        per_modality=per_modality,
    )


def format_report(report: AblationReport) -> str:
    lines = [f"duplicate-detection ablation @K={report.k}", f"dataset={report.dataset_version} model={report.model_version}"]
    for m in ("text", "image", "multimodal"):
        e = report.per_modality[m]
        lines.append(f"{m}: recall@{e.k}={e.recall_at_k:.3f} mrr={e.mrr:.3f} map={e.map:.3f} (n={e.n_queries})")
    return "\n".join(lines)

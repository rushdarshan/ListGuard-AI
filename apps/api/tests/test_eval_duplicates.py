"""Duplicate-detection eval pipeline — RED tests (written BEFORE implementation).

Metric values are hand-computed below. Retrieval comparisons use the real
app.retrieval fixture embeddings; tests assert measured properties
(versioning, ranges, ordering) — never invented numbers.
"""

import pytest

from app.eval_duplicates import (
    AblationReport,
    EvalDataset,
    EvalMetrics,
    EvalQuery,
    ablation_report,
    average_precision,
    evaluate_modality,
    evaluate_rankings,
    format_report,
    reciprocal_rank,
    recall_at_k,
    retrieve_ranked_ids,
)


def _tiny_docs():
    return [
        {"id": "bike-red", "title": "Red bicycle fast road bike", "description": "", "image_sha256": "a" * 64},
        {"id": "kayak-blue", "title": "Blue kayak river boat", "description": "", "image_sha256": "b" * 64},
        {"id": "bike-red-2", "title": "Red bicycle mountain trail", "description": "", "image_sha256": "a" * 64},
    ]


def _tiny_dataset():
    return EvalDataset(
        version="dup-eval-v1",
        queries=[
            EvalQuery(query_id="q1", text="red bicycle", relevant_ids=["bike-red", "bike-red-2"], image_sha256="a" * 64),
            EvalQuery(query_id="q2", text="blue kayak", relevant_ids=["kayak-blue"], image_sha256="b" * 64),
        ],
    )


# ---- metric correctness (hand-computed) ----


def test_recall_at_k_hand_computed():
    assert recall_at_k(["a", "b"], ["a", "x", "b"], 2) == pytest.approx(0.5)
    assert recall_at_k(["a"], ["x", "a"], 1) == pytest.approx(0.0)
    assert recall_at_k(["a"], ["x", "a"], 2) == pytest.approx(1.0)


def test_recall_at_k_empty_relevant_is_zero():
    assert recall_at_k([], ["a"], 5) == pytest.approx(0.0)
    assert recall_at_k(["a"], [], 5) == pytest.approx(0.0)


def test_recall_at_k_invalid_k_raises():
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], 0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], -1)


def test_reciprocal_rank_hand_computed():
    assert reciprocal_rank(["a"], ["a", "b"]) == pytest.approx(1.0)
    assert reciprocal_rank(["a"], ["x", "a"]) == pytest.approx(0.5)
    assert reciprocal_rank(["a"], ["x", "y"]) == pytest.approx(0.0)
    assert reciprocal_rank([], ["a"]) == pytest.approx(0.0)


def test_average_precision_hand_computed():
    # hits at ranks 1,3 over 2 relevant: (1/1 + 2/3) / 2
    assert average_precision(["a", "b"], ["a", "x", "b"]) == pytest.approx((1.0 + 2 / 3) / 2)
    assert average_precision(["a"], ["x", "a"]) == pytest.approx(0.5)
    assert average_precision(["a"], ["x"]) == pytest.approx(0.0)
    assert average_precision([], ["a"]) == pytest.approx(0.0)


def test_evaluate_rankings_averages_over_queries():
    ds = EvalDataset(
        version="v1",
        queries=[
            EvalQuery(query_id="q1", text="t", relevant_ids=["a"]),
            EvalQuery(query_id="q2", text="t", relevant_ids=["b"]),
        ],
    )
    r, mrr, m = evaluate_rankings({"q1": ["a"], "q2": ["x", "b"]}, ds, k=2)
    assert r == pytest.approx(1.0)  # both relevant ids surface in top-2
    assert mrr == pytest.approx((1.0 + 0.5) / 2)
    assert m == pytest.approx((1.0 + 0.5) / 2)


def test_evaluate_rankings_empty_dataset_raises():
    with pytest.raises(ValueError):
        evaluate_rankings({}, EvalDataset(version="v1", queries=[]), k=5)


# ---- retrieval modality comparison (real code, measured properties) ----


def test_retrieve_ranked_ids_text_finds_bike():
    docs = _tiny_docs()
    q = EvalQuery(query_id="q1", text="red bicycle", relevant_ids=["bike-red"])
    ranked = retrieve_ranked_ids(q, docs, "text", top_k=3)
    assert ranked[0] in ("bike-red", "bike-red-2")
    assert set(ranked) == {"bike-red", "bike-red-2"}  # text search returns overlap hits only


def test_retrieve_ranked_ids_multimodal_ranks_all_docs():
    docs = _tiny_docs()
    q = _tiny_dataset().queries[0]
    ranked = retrieve_ranked_ids(q, docs, "multimodal", top_k=3)
    assert set(ranked) == {"bike-red", "kayak-blue", "bike-red-2"}
    assert ranked[0] in ("bike-red", "bike-red-2")


def test_evaluate_modality_empty_dataset_raises():
    with pytest.raises(ValueError):
        evaluate_modality(EvalDataset(version="v", queries=[]), _tiny_docs(), "text", k=2)


def test_retrieve_ranked_ids_unknown_modality_raises():
    with pytest.raises(ValueError):
        retrieve_ranked_ids(_tiny_dataset().queries[0], _tiny_docs(), "audio", top_k=3)  # type: ignore[arg-type]


def test_retrieve_ranked_ids_image_without_sha_raises():
    q = EvalQuery(query_id="q", text="bike", relevant_ids=["bike-red"])  # no image_sha256
    with pytest.raises(ValueError):
        retrieve_ranked_ids(q, _tiny_docs(), "image", top_k=3)


def test_evaluate_modality_versions_and_ranges():
    ds, docs = _tiny_dataset(), _tiny_docs()
    for mod in ("text", "image", "multimodal"):
        m = evaluate_modality(ds, docs, mod, k=2, model_version="m1")  # type: ignore[arg-type]
        assert isinstance(m, EvalMetrics)
        assert m.dataset_version == "dup-eval-v1" and m.model_version == "m1" and m.k == 2
        assert m.n_queries == 2
        assert 0.0 <= m.recall_at_k <= 1.0 and 0.0 <= m.mrr <= 1.0 and 0.0 <= m.map <= 1.0


def test_evaluate_modality_text_beats_nonsense_query():
    docs = _tiny_docs()
    good = EvalDataset(version="v", queries=[EvalQuery(query_id="q", text="red bicycle", relevant_ids=["bike-red", "bike-red-2"])])
    bad = EvalDataset(version="v", queries=[EvalQuery(query_id="q", text="zzzqqq", relevant_ids=["bike-red", "bike-red-2"])])
    assert evaluate_modality(good, docs, "text", k=3).recall_at_k >= evaluate_modality(bad, docs, "text", k=3).recall_at_k


def test_ablation_report_covers_all_modalities():
    report = ablation_report(_tiny_dataset(), _tiny_docs(), k=2, model_version="m1")
    assert isinstance(report, AblationReport)
    assert set(report.per_modality) == {"text", "image", "multimodal"}
    assert report.dataset_version == "dup-eval-v1" and report.model_version == "m1"


def test_format_report_contains_versions_and_numbers():
    report = ablation_report(_tiny_dataset(), _tiny_docs(), k=2, model_version="m1")
    text = format_report(report)
    assert "dup-eval-v1" in text and "m1" in text
    for mod in ("text", "image", "multimodal"):
        assert mod in text


def test_ablation_report_survives_text_only_dataset():
    docs = [{"id": "d1", "title": "Red bicycle", "description": ""}]
    ds = EvalDataset(version="v-text", queries=[
        EvalQuery(query_id="q1", text="red bike", relevant_ids=["d1"])])
    report = ablation_report(ds, docs, k=2, model_version="m1")
    assert set(report.per_modality) == {"text", "image", "multimodal"}
    assert report.per_modality["image"].n_queries == 0  # unevaluated, never fabricated
    assert report.per_modality["text"].n_queries == 1

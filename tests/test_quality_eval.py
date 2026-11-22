"""Phase 2.6: deterministic P@K, page recall, MRR, suite integrity and reports."""
import json
from pathlib import Path

import pytest

from papertrail.retrieval import RetrievalError
from papertrail.retrieval.quality_eval import (
    RankedPage, RetrievalSuite, load_retrieval_suite, parse_k_values,
    render_markdown_report, run_retrieval_evaluation, save_retrieval_report,
    score_at_k,
)
from test_hybrid_retrieval import corpus  # noqa: F401; shared synthetic corpus fixture
from test_vector_search import FakeEncoder, FakeFaiss, PAPER


def hits(*pages):
    return [RankedPage(rank=i, page_number=page, chunk_id=f"c{i}")
            for i, page in enumerate(pages, 1)]


def suite_file(tmp_path: Path, paper_id=PAPER, items=None):
    if items is None:
        items = [
            {"id": "benchmark_names", "question": "Which benchmark datasets were chosen?",
             "labels": [{"page_number": 1, "note": "Named multiple research QA benchmarks."}]},
            {"id": "assessment", "question": "How is structured evidence assessed?",
             "labels": [{"page_number": 2, "note": "Describes the evidence assessment component."}]},
        ]
    payload = {
        "schema_version": 1, "suite_id": "fixture_suite_v1", "paper_id": paper_id,
        "annotation_status": "provisional", "annotation_unit": "page",
        "provenance": "Offline synthetic fixture for automated test coverage, not a real research judgment.",
        "cases": items,
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_scoring_known_ranking_at_multiple_k():
    data = hits(2, 2, 3, 4)
    at1 = score_at_k(data, {2, 4}, 1)
    assert at1.precision_at_k == .999999 or at1.precision_at_k == 1
    assert at1.recall_at_k == .5 and at1.mrr_at_k == 1.0
    at3 = score_at_k(data, {2, 4}, 3)
    assert at3.precision_at_k == pytest.approx(2 / 3, abs=1e-6)
    assert at3.recall_at_k == .5  # duplicate primary hits on page 2 do not raise recall
    assert at3.relevant_hits == 2 and at3.distinct_relevant_pages == 1
    at4 = score_at_k(data, {2, 4}, 4)
    assert at4.precision_at_k == .75 and at4.recall_at_k == 1
    assert at4.mrr_at_k == 1 and at4.hit_at_k == 1


def test_first_relevant_rank_and_missing_slots():
    at4 = score_at_k(hits(1, 3, 8), {8}, 4)
    assert at4.precision_at_k == .25
    assert at4.recall_at_k == 1
    assert at4.mrr_at_k == pytest.approx(.333333, abs=1e-6)
    assert at4.returned_hits == 3
    missing = score_at_k(hits(1, 3), {8}, 5)
    assert (missing.precision_at_k, missing.recall_at_k, missing.mrr_at_k, missing.hit_at_k) == (0, 0, 0, 0)
    empty = score_at_k([], {1}, 3)
    assert empty.returned_hits == 0 and empty.hit_at_k == 0


@pytest.mark.parametrize("bad", ["", "1,", "3,1", "1,1", "0,3", "1,21", "x", "1 3", [True, 2], [1, 1]])
def test_invalid_k_values(bad):
    with pytest.raises(RetrievalError, match="k-values"):
        parse_k_values(bad)


def test_k_parse_accepts_ordered_values():
    assert parse_k_values(" 1, 3, 5, 10 ") == (1, 3, 5, 10)


def test_suite_missing_and_malformed(tmp_path):
    with pytest.raises(RetrievalError, match="not found"):
        load_retrieval_suite(tmp_path / "absent.json")
    path = tmp_path / "suite.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(RetrievalError, match="Invalid"):
        load_retrieval_suite(path)
    path.write_text(" " * 100001, encoding="utf-8")
    with pytest.raises(RetrievalError, match="size limit"):
        load_retrieval_suite(path)


def test_suite_case_id_and_labels_must_be_unique(tmp_path):
    base = {"id": "duplicate", "question": "Where does the answer appear?",
            "labels": [{"page_number": 1, "note": "Cited evidence from source page one."}]}
    with pytest.raises(RetrievalError, match="Duplicate query identifiers"):
        load_retrieval_suite(suite_file(tmp_path, items=[base, base]))
    duplicated_pages = {**base, "labels": base["labels"] * 2}
    with pytest.raises(RetrievalError, match="Duplicate relevant pages"):
        load_retrieval_suite(suite_file(tmp_path, items=[duplicated_pages]))


def test_original_suite_is_provisional_and_grounded_to_prior_logs():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "fair_rag_pages_v1.json"
    suite = load_retrieval_suite(path)
    assert isinstance(suite, RetrievalSuite)
    assert suite.paper_id == "2510.22344"
    assert suite.annotation_status == "provisional"
    assert suite.annotation_unit == "page"
    assert len(suite.cases) == 8
    assert all(c.labels for c in suite.cases)


def evaluate_synthetic(corpus, suite):
    return run_retrieval_evaluation(
        PAPER, data_dir=corpus, suite_path=suite, k_values="1,3,5,10",
        encoder_factory=FakeEncoder, faiss_backend=FakeFaiss,
    )


def test_full_evaluation_three_methods_and_aggregates(corpus, tmp_path):
    report = evaluate_synthetic(corpus, suite_file(tmp_path))
    assert report.query_count == 2
    assert report.k_values == [1, 3, 5, 10]
    assert report.methods == ["bm25", "faiss", "hybrid"]
    assert report.annotation_status == "provisional"
    assert report.corpus_chunk_sha256 and len(report.corpus_chunk_sha256) == 64
    assert not report.pdf_extraction_quality_verified
    assert len(report.query_results) == 2
    for case in report.query_results:
        for method in report.methods:
            assert len(case.top_hits[method]) <= 10
            assert len({row.chunk_id for row in case.top_hits[method]}) == len(case.top_hits[method])
            for k in report.k_values:
                metric = case.metrics[method][str(k)]
                assert 0 <= metric.precision_at_k <= 1
                assert 0 <= metric.recall_at_k <= 1
                assert metric.relevant_hits <= min(k, len(case.top_hits[method]))
    for method in report.methods:
        for k in report.k_values:
            expected = sum(case.metrics[method][str(k)].mrr_at_k for case in report.query_results) / 2
            assert report.macro_results[method][str(k)].mrr_at_k == pytest.approx(expected, abs=1e-6)


def test_reports_are_deterministic_and_do_not_dump_full_paper(corpus, tmp_path):
    report = evaluate_synthetic(corpus, suite_file(tmp_path))
    report2 = evaluate_synthetic(corpus, suite_file(tmp_path))
    assert report.model_dump_json() == report2.model_dump_json()
    jpath, mpath = save_retrieval_report(report, data_dir=tmp_path / "reports")
    assert jpath.exists() and mpath.exists()
    payload = json.loads(jpath.read_text(encoding="utf-8"))
    assert payload["query_count"] == 2
    assert payload["macro_results"]["hybrid"]["3"]["mrr_at_k"] >= 0
    md = mpath.read_text(encoding="utf-8")
    assert "MRR@K" in md and "Precision@K" in md and "Recall@K" in md
    assert "page-proxy" in md and "provisional" in md
    assert "The Structured Evidence Assessment agent inspects missing information." not in md
    assert render_markdown_report(report) == md


def test_mismatched_paper_or_missing_relevance_pages_fail_closed(corpus, tmp_path):
    path = suite_file(tmp_path, paper_id="9999.99999")
    with pytest.raises(RetrievalError, match="does not match"):
        evaluate_synthetic(corpus, path)
    path = suite_file(tmp_path, items=[{"id": "unobserved", "question": "Which page has the evidence?",
        "labels": [{"page_number": 99, "note": "Source page is not represented by chunks."}]}])
    with pytest.raises(RetrievalError, match="missing from saved chunks"):
        evaluate_synthetic(corpus, path)


def test_one_encoder_instance_reused_for_all_methods_and_questions(corpus, tmp_path):
    calls = []

    def encoder(name, revision):
        calls.append((name, revision))
        return FakeEncoder(name, revision)

    report = run_retrieval_evaluation(
        PAPER, data_dir=corpus, suite_path=suite_file(tmp_path), k_values="1,3",
        encoder_factory=encoder, faiss_backend=FakeFaiss,
    )
    assert report.query_count == 2
    assert len(calls) == 1

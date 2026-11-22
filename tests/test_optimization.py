"""Leakage-controlled holdout optimization with mocked FAISS and real cached chunks."""
import json
import hashlib
from pathlib import Path

import pytest

from papertrail.retrieval import RetrievalError
from papertrail.retrieval import optimization
from papertrail.retrieval.optimization import (
    CANDIDATE_GRID, _split_queries, run_retrieval_optimization,
    render_optimization_markdown, save_optimization_report,
)
from test_hybrid_retrieval import corpus  # noqa: F401
from test_vector_search import FakeFaiss, FakeEncoder, PAPER


CASES = [
    ('dataset_one', 'Which multi-hop benchmark datasets were evaluated?', 1),
    ('dataset_two', 'Where is the complex MuSiQue evidence?', 1),
    ('assessment', 'How does Structured Evidence Assessment find missing evidence?', 2),
    ('trivia_fact', 'Which factual TriviaQA benchmark was evaluated?', 2),
    ('benchmark_extra', 'What are the multiple evaluation benchmarks?', 1),
    ('assessment_extra', 'Where is the Structured Evidence Assessment module described?', 2),
    ('another_question', 'What is the evidence process for question answering?', 2),
    ('another_benchmark', 'Which datasets are multi-hop tasks?', 1),
]


def build_suite(tmp_path, cases=CASES):
    path = tmp_path / 'benchmark.json'
    data = {
        'schema_version': 1, 'suite_id': 'synthetic_holdout_v1', 'paper_id': PAPER,
        'annotation_status': 'provisional', 'annotation_unit': 'page',
        'provenance': 'Synthetic test-only source labels deliberately separate from production usage.',
        'cases': [{'id': name, 'question': question, 'labels': [
            {'page_number': page, 'note': 'Synthetic page attribution for deterministic tests.'}
        ]} for name, question, page in cases],
    }
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def run(corpus, path, **kw):
    return run_retrieval_optimization(PAPER, data_dir=corpus, suite_path=path,
        encoder_factory=FakeEncoder, faiss_backend=FakeFaiss, **kw)


def test_optimizer_holdout_and_report_integrity(corpus, tmp_path):
    suite = build_suite(tmp_path)
    report = run(corpus, suite)
    assert report.vector_count > 1
    assert report.candidate_collection_count == 8
    assert (len(report.train_query_ids), len(report.validation_query_ids)) == (6, 2)
    assert not (set(report.train_query_ids) & set(report.validation_query_ids))
    assert set(report.train_query_ids + report.validation_query_ids) == {c[0] for c in CASES}
    assert report.suite_sha256 == hashlib.sha256(suite.read_bytes()).hexdigest()
    assert report.annotation_status == 'provisional'
    assert report.pdf_extraction_quality_verified is False
    assert len(report.grid_results) == len(CANDIDATE_GRID)
    assert 'no_auto_deploy' in report.deployment_recommendation or 'keep_default' in report.deployment_recommendation
    assert 0 <= report.selected_validation_score <= 1
    assert 'manual' in report.methodology_note.lower()
    assert 0 <= report.candidate_collection_count <= len(CASES)
    assert all(r.dense_weight >= 0 and r.bm25_weight >= 0 for r in report.grid_results)
    jpath, mpath = save_optimization_report(report, data_dir=tmp_path / 'out')
    assert json.loads(jpath.read_text(encoding='utf-8'))['annotation_status'] == 'provisional'
    assert mpath.read_text(encoding='utf-8') == render_optimization_markdown(report)
    assert 'The Structured Evidence Assessment agent inspects' not in mpath.read_text()


def test_query_split_deterministic_and_order_independent():
    ids = [x[0] for x in CASES]
    assert _split_queries(ids, 42) == _split_queries(list(reversed(ids)), 42)
    a, b = _split_queries(ids, 42)
    assert len(a) == 6 and len(b) == 2
    assert set(a).isdisjoint(b)


def test_report_repeatable_on_same_corpus(corpus, tmp_path):
    suite = build_suite(tmp_path)
    report = run(corpus, suite)
    assert report.model_dump_json() == run(corpus, suite).model_dump_json()


def test_optimization_selection_does_not_use_validation_labels(corpus, tmp_path):
    suite = build_suite(tmp_path)
    first = run(corpus, suite)
    holdout = set(first.validation_query_ids)
    modified = [(name, question, (1 if page == 2 else 2) if name in holdout else page)
                for name, question, page in CASES]
    changed = run(corpus, build_suite(tmp_path, modified))
    assert first.train_query_ids == changed.train_query_ids
    assert (first.selected_dense_weight, first.selected_bm25_weight) == (
        changed.selected_dense_weight, changed.selected_bm25_weight)
    assert [x.train_score for x in first.grid_results] == [x.train_score for x in changed.grid_results]


def test_reports_unlabeled_pages_as_review_candidates_not_auto_labels(corpus, tmp_path):
    report = run(corpus, build_suite(tmp_path))
    assert report.flagged_unlabeled_pages
    for item in report.flagged_unlabeled_pages:
        assert 'MANUAL' in item.note
        assert item.retrieval_methods
        labels = {page for name, _, page in CASES if name == item.case_id}
        assert item.page_number not in labels


def test_models_encoded_once_per_query_not_once_per_grid(corpus, tmp_path, monkeypatch):
    calls = []
    original = optimization.search_paper

    def monitor(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(optimization, 'search_paper', monitor)
    run(corpus, build_suite(tmp_path))
    assert len(calls) == len(CASES)
    assert len(set(calls)) == len(CASES)


@pytest.mark.parametrize('metric', ['junk', '', None])
def test_invalid_metrics(corpus, tmp_path, metric):
    with pytest.raises(RetrievalError, match='metric'):
        run(corpus, build_suite(tmp_path), metric=metric)


@pytest.mark.parametrize('k', [0, 21, -1, True, 2.5])
def test_invalid_k(corpus, tmp_path, k):
    with pytest.raises(RetrievalError, match='k must'):
        run(corpus, build_suite(tmp_path), k=k)


@pytest.mark.parametrize('seed', [-1, 2**32, True, '42'])
def test_invalid_seed(corpus, tmp_path, seed):
    with pytest.raises(RetrievalError, match='seed'):
        run(corpus, build_suite(tmp_path), seed=seed)


def test_suite_mismatch_rejected_before_search(corpus, tmp_path):
    path = build_suite(tmp_path)
    data = json.loads(path.read_text())
    data['paper_id'] = '9999.99999'
    path.write_text(json.dumps(data))
    with pytest.raises(RetrievalError, match='does not match'):
        run(corpus, path)


def test_tiny_suite_rejected(corpus, tmp_path):
    with pytest.raises(RetrievalError, match='at least four'):
        run(corpus, build_suite(tmp_path, CASES[:3]))


def test_missing_label_page_fails_closed(corpus, tmp_path):
    cases = [*CASES[:3], ('invalid_page', 'Where is the missing evidence page?', 99)]
    with pytest.raises(RetrievalError, match='absent'):
        run(corpus, build_suite(tmp_path, cases))

"""CLI smoke contract: offline labels cannot auto-change production weights."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from papertrail import cli
from papertrail.retrieval.faiss_store import RetrievalError


def test_optimize_cli_json(monkeypatch, capsys):
    def fake(paper_id, *, data_dir, suite_path, seed, metric, k):
        assert paper_id == '2510.22344'
        assert (seed, metric, k) == (11, 'mrr_at_k', 3)
        assert suite_path == Path('benchmarks/fair_rag_pages_v1.json')
        return SimpleNamespace(
            paper_id=paper_id, suite_id='fair_rag_pages_v1', annotation_status='provisional',
            metric=metric, k=k, train_query_ids=['q1','q2'], validation_query_ids=['q3','q4'],
            selected_dense_weight=1., selected_bm25_weight=2., selected_page_cap=0,
            baseline_validation_score=.2, selected_validation_score=.4,
            bm25_validation_score=.3, faiss_validation_score=.1,
            deployment_recommendation='exploratory_gain_only_manual_review_required_no_auto_deploy',
            flagged_unlabeled_pages=[],
        )
    monkeypatch.setattr(cli, 'run_retrieval_optimization', fake)
    monkeypatch.setattr(cli, 'save_optimization_report', lambda r, *, data_dir: (Path('a.json'), Path('b.md')))
    assert cli.main(['optimize-retrieval','--paper-id','2510.22344','--seed','11','--k','3']) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj['status'] == 'evaluated_not_deployed'
    assert obj['selected_bm25_weight'] == 2
    assert obj['pdf_extraction_quality_verified'] is False


def test_weighted_hybrid_cli_passes_requested_weights(monkeypatch, capsys):
    def fake(*args, **kwargs):
        assert (kwargs['dense_weight'], kwargs['bm25_weight']) == (0.5, 3.)
        return SimpleNamespace(model_dump=lambda **kwargs: {'hits': [], 'dense_weight': 0.5})
    monkeypatch.setattr(cli, 'hybrid_search_paper', fake)
    assert cli.main(['hybrid-search','--paper-id','2510.22344','--query','test','--dense-weight','0.5','--bm25-weight','3']) == 0
    assert json.loads(capsys.readouterr().out)['dense_weight'] == .5


def test_cli_optimizer_validation_error_safe(monkeypatch, capsys):
    monkeypatch.setattr(cli,'run_retrieval_optimization', lambda *a, **k: (_ for _ in ()).throw(RetrievalError('No embeddings')))
    assert cli.main(['optimize-retrieval','--paper-id','2510.22344']) == 2
    assert 'No embeddings' in capsys.readouterr().err

"""New CLI contracts, no SentenceTransformer or FAISS installs required."""
import json
from types import SimpleNamespace

from papertrail.cli import main
from papertrail.retrieval import RetrievalError


def test_hybrid_command(monkeypatch, capsys):
    def fake(paper_id, query, *, data_dir, top_k, context_window, dense_weight, bm25_weight, max_hits_per_page):
        assert max_hits_per_page == 0
        assert (dense_weight, bm25_weight) == (1.0, 1.0)
        assert (paper_id, query, top_k, context_window) == ("2510.22344", "benchmark", 7, 2)
        return SimpleNamespace(model_dump=lambda **kwargs: {
            "method": "hybrid_rrf", "hits": [{"page_number": 9, "score": 0.03}]
        })
    monkeypatch.setattr("papertrail.cli.hybrid_search_paper", fake)
    assert main(["hybrid-search", "--paper-id", "2510.22344", "--query", "benchmark",
                 "--top-k", "7", "--context-window", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["method"] == "hybrid_rrf"


def test_comparison_command(monkeypatch, capsys):
    def fake(paper_id, *, data_dir, top_k, context_window):
        assert (paper_id, top_k, context_window) == ("2510.22344", 10, 1)
        return SimpleNamespace(model_dump=lambda **kwargs: {
            "dense": {"covered_count": 2}, "hybrid": {"covered_count": 4}
        })
    monkeypatch.setattr("papertrail.cli.compare_benchmark_retrieval", fake)
    assert main(["evaluate-benchmarks", "--paper-id", "2510.22344"]) == 0
    assert json.loads(capsys.readouterr().out)["hybrid"]["covered_count"] == 4


def test_hybrid_cli_errors_safe(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RetrievalError("Run embed-document first")
    monkeypatch.setattr("papertrail.cli.hybrid_search_paper", fail)
    assert main(["hybrid-search", "--paper-id", "2510.22344", "--query", "test"]) == 2
    assert "Run embed-document first" in capsys.readouterr().err

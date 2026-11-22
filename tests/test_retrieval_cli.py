"""CLI contract tests without native FAISS or a downloaded embedding model."""
import json
from pathlib import Path
from types import SimpleNamespace

from papertrail.cli import main
from papertrail.retrieval import RetrievalError


def test_index_document_command(monkeypatch, capsys, tmp_path):
    info = SimpleNamespace(paper_id="2510.22344", vector_count=365,
                           dimension=384, index_type="IndexFlatIP", metric="inner_product")
    def fake_build(paper_id, *, data_dir, force):
        assert paper_id == "2510.22344" and force is True
        return SimpleNamespace(manifest=info, cached=False,
                               index_path=Path("data/indexes/2510.22344.faiss"),
                               manifest_path=Path("data/indexes/2510.22344.manifest.json"))
    monkeypatch.setattr("papertrail.cli.build_paper_index", fake_build)
    assert main(["index-document", "--paper-id", "2510.22344", "--force"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "indexed" and result["vector_count"] == 365
    assert result["metric"] == "inner_product"


def test_search_chunks_command(monkeypatch, capsys):
    def fake_search(paper_id, query, *, data_dir, top_k):
        assert paper_id == "2510.22344"
        assert query == "query about SEA" and top_k == 2
        return SimpleNamespace(model_dump=lambda **kwargs: {
            "paper_id": paper_id, "query": query,
            "hits": [{"score": 0.8, "citation": "https://arxiv.org/abs/2510.22344#page=2"}],
        })
    monkeypatch.setattr("papertrail.cli.search_paper", fake_search)
    assert main(["search-chunks", "--paper-id", "2510.22344", "--query", "query about SEA", "--top-k", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["hits"][0]["score"] == 0.8


def test_retrieval_cli_error_is_actionable(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RetrievalError("Run embed-document first")
    monkeypatch.setattr("papertrail.cli.build_paper_index", fail)
    assert main(["index-document", "--paper-id", "2510.22344"]) == 2
    assert "Run embed-document first" in capsys.readouterr().err

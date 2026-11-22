"""Integration of a new CLI command with report saving and error contracts."""
import json
from pathlib import Path

from papertrail import cli
from papertrail.retrieval.quality_eval import (
    MetricSummary, RetrievalEvaluationReport,
)


def fake_report():
    return RetrievalEvaluationReport(
        suite_id="fixture_suite_v1", paper_id="2510.22344", annotation_status="provisional",
        corpus_chunk_sha256="a" * 64, vector_count=10, embedding_model="mock", model_revision="v1",
        query_count=1, k_values=[3], methods=["bm25", "faiss", "hybrid"],
        macro_results={name: {"3": MetricSummary(precision_at_k=.333333, recall_at_k=1,
                         mrr_at_k=1, hit_at_k=1)} for name in ["bm25", "faiss", "hybrid"]},
        query_results=[],
    )


def test_cli_evaluate_retrieval_json_only(monkeypatch, capsys, tmp_path):
    def evaluate(paper_id, *, data_dir, suite_path, k_values, dense_weight, bm25_weight, max_hits_per_page):
        assert max_hits_per_page == 0
        assert (dense_weight, bm25_weight) == (1.0, 1.0)
        assert paper_id == "2510.22344"
        assert suite_path == Path("some_suite.json")
        assert k_values == "3"
        return fake_report()
    def save(report, *, data_dir):
        assert report.suite_id == "fixture_suite_v1"
        return (Path("data/evaluations/test.json"), Path("data/evaluations/test.md"))
    monkeypatch.setattr(cli, "run_retrieval_evaluation", evaluate)
    monkeypatch.setattr(cli, "save_retrieval_report", save)
    assert cli.main(["evaluate-retrieval", "--paper-id", "2510.22344", "--suite", "some_suite.json", "--k-values", "3"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "evaluated"
    assert output["questions"] == 1
    assert output["macro_results"]["hybrid"]["3"]["recall_at_k"] == 1
    assert output["pdf_extraction_quality_verified"] is False


def test_cli_missing_suite_fails_without_traceback(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["evaluate-retrieval", "--paper-id", "2510.22344", "--suite", "missing.json"]) == 2
    err = capsys.readouterr().err
    assert "Evaluation suite not found" in err and "Traceback" not in err

"""Phase 4.1 offline stage diagnostics: no new live LLM calls, no gold labels."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from papertrail.agent.evaluation import (
    AgentEvaluationError, AgentEvaluationSuite, evaluate_preview_run, load_agent_suite,
    load_recorded_live_snapshot, parse_k_values, run_agent_evaluation, save_agent_evaluation,
)

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/agent_fairrag_log_anchors_v1.json"
REPLAY = ROOT / "benchmarks/agent_fairrag_observed_live_v1.json"


def mock_preview(suite, k):
    case = suite.cases[0]
    evidence = []
    register = []
    for facet in case.facets:
        retrieved = [{"chunk_id": a} for a in facet.anchor_chunk_ids]
        retrieved.extend({"chunk_id": f"other_{facet.step_id}_{n}"} for n in range(k))
        evidence.append({"step_id": facet.step_id, "hits": retrieved[:k]})
        for i, hit in enumerate(retrieved[:min(k, 2)]):
            register.append({"step_id": facet.step_id, "chunk_id": hit["chunk_id"],
                             "evidence_id": f"E{len(register)+1:02d}"})
    return {"paper_id": case.paper_id, "llm_calls": 0, "network_opt_in": False,
            "generation_mode": "preview", "answer_generated": False,
            "status": "ready_for_llm_synthesis", "evidence": evidence,
            "synthesis": {"source_register": register}}


def test_suite_is_explicitly_provisional():
    suite = load_agent_suite(SUITE)
    assert suite.annotation_status == "provisional_log_derived_unverified"
    assert suite.cases[0].paper_id == "2510.22344"
    assert [facet.step_id for facet in suite.cases[0].facets] == ["retrieve_01", "retrieve_02", "retrieve_03"]


def test_stage_metrics_separate_retrieval_from_selection():
    suite = load_agent_suite(SUITE)
    result = mock_preview(suite, 3)
    facet3 = suite.cases[0].facets[2]
    # The label is reached by retrieval, but excluded from the model-visible register.
    result["synthesis"]["source_register"] = [r for r in result["synthesis"]["source_register"]
                                                   if not (r["step_id"] == facet3.step_id and
                                                           r["chunk_id"] == facet3.anchor_chunk_ids[0])]
    report = evaluate_preview_run(result, suite.cases[0], 3)
    assert report["anchors_retrieved"] == 4
    assert report["anchors_model_visible"] == 3
    assert report["facets_with_observed_retrieval_anchor"] == 3
    assert report["facets"][2]["retrieved_but_not_selected_ids"] == [facet3.anchor_chunk_ids[0]]
    assert report["facets"][2]["not_retrieved_ids"] == []


def test_anchor_absent_from_retrieval_is_not_labelled_as_selection_failure():
    suite = load_agent_suite(SUITE)
    result = mock_preview(suite, 3)
    facet3 = suite.cases[0].facets[2]
    result["evidence"][2]["hits"] = [{"chunk_id": "other"}]
    result["synthesis"]["source_register"] = [r for r in result["synthesis"]["source_register"]
                                                   if r["step_id"] != facet3.step_id]
    row = evaluate_preview_run(result, suite.cases[0], 3)["facets"][2]
    assert row["not_retrieved_ids"] == facet3.anchor_chunk_ids
    assert row["retrieved_but_not_selected_ids"] == []


@pytest.mark.parametrize("k_values", ["", "3,3", "5,3", "0", "11", "1,2,3,4,5", "3,", "1;2", "1,2,3,4,5,6,7"])
def test_bad_k_values_rejected(k_values):
    with pytest.raises(AgentEvaluationError):
        parse_k_values(k_values)


def test_k_values_accepted():
    assert parse_k_values("3,5") == (3, 5)
    assert parse_k_values("1") == (1,)


def test_invalid_suite_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    obj = json.loads(SUITE.read_text(encoding="utf8"))
    obj["annotation_status"] = "independently_verified_gold"
    bad.write_text(json.dumps(obj), encoding="utf8")
    with pytest.raises(AgentEvaluationError):
        load_agent_suite(bad)


def test_suite_chunk_ids_must_be_unique_and_safe(tmp_path):
    obj = json.loads(SUITE.read_text(encoding="utf8"))
    obj["cases"][0]["facets"][0]["anchor_chunk_ids"] = ["../../../../../secrets"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(obj), encoding="utf8")
    with pytest.raises(AgentEvaluationError):
        load_agent_suite(bad)


def test_oversized_suite_rejected(tmp_path):
    big = tmp_path / "huge.json"
    big.write_bytes(b" " * 256_001)
    with pytest.raises(AgentEvaluationError):
        load_agent_suite(big)


def test_llm_and_network_guards():
    suite = load_agent_suite(SUITE)
    result = mock_preview(suite, 3)
    result["llm_calls"] = 1
    with pytest.raises(AgentEvaluationError, match="Offline evaluation invariant"):
        evaluate_preview_run(result, suite.cases[0], 3)
    result["llm_calls"] = 0
    result["network_opt_in"] = True
    with pytest.raises(AgentEvaluationError, match="Offline evaluation invariant"):
        evaluate_preview_run(result, suite.cases[0], 3)
    result["network_opt_in"] = False
    result["answer_generated"] = True
    with pytest.raises(AgentEvaluationError, match="non-generating"):
        evaluate_preview_run(result, suite.cases[0], 3)


def test_selection_cannot_claim_unretrieved_anchor():
    suite = load_agent_suite(SUITE)
    result = mock_preview(suite, 3)
    result["evidence"][0]["hits"] = [{"chunk_id": "nope"}]
    with pytest.raises(AgentEvaluationError, match="not from retrieved"):
        evaluate_preview_run(result, suite.cases[0], 3)


def test_incomplete_run_refused():
    suite = load_agent_suite(SUITE)
    result = mock_preview(suite, 3)
    result["status"] = "tool_failed"
    with pytest.raises(AgentEvaluationError):
        evaluate_preview_run(result, suite.cases[0], 3)


def test_failed_native_preview_reports_sanitized_stage_and_k():
    suite = load_agent_suite(SUITE)
    result = mock_preview(suite, 3)
    result.update({
        "status": "tool_failed",
        "tool_events": [
            {"name": "index_document", "status": "success", "detail": {}},
            {"name": "hybrid_retrieval", "status": "failed",
             "detail": {"error_type": "MemoryError", "provider_response": "DO_NOT_DISPLAY"}},
        ],
        "trace": [{"node": "retrieve", "event": "tool_failed"}],
        "errors": ["Private path or API secret must not leak"],
    })
    with pytest.raises(AgentEvaluationError) as exc:
        evaluate_preview_run(result, suite.cases[0], 3)
    message = str(exc.value)
    assert "top_k=3" in message
    assert "case=fairrag_multifacet_probe" in message
    assert "status=tool_failed" in message
    assert "failed_tool=hybrid_retrieval" in message
    assert "error_type=MemoryError" in message
    assert "last_node=retrieve" in message
    assert "DO_NOT_DISPLAY" not in message
    assert "Private path" not in message


def test_untrusted_tool_fields_are_not_redisplayed():
    suite = load_agent_suite(SUITE)
    result = mock_preview(suite, 5)
    result.update({
        "status": "malicious prompt with secret=abc",
        "tool_events": [{"name": "tool secret=abc", "status": "failed",
                          "detail": {"error_type": "abc secret=abc"}}],
        "trace": [{"node": "secret=abc"}],
    })
    with pytest.raises(AgentEvaluationError) as exc:
        evaluate_preview_run(result, suite.cases[0], 5)
    message = str(exc.value)
    assert "secret=abc" not in message
    assert "top_k=5" in message
    assert "status=unknown" in message
    assert "failed_tool=unknown" in message


def test_historical_replay_unverified_and_not_new_call():
    replay = load_recorded_live_snapshot(REPLAY, "2510.22344")
    assert replay["accepted_claims"] == 4
    assert replay["rejected_claims"] == 2
    assert replay["observed_claim_acceptance_fraction"] == 0.6667
    assert replay["observed_facet_coverage_fraction"] == 0.6667
    assert replay["independent_claim_validation_performed"] is False


def test_historical_replay_cannot_claim_semantic_validation(tmp_path):
    obj = json.loads(REPLAY.read_text(encoding="utf8"))
    obj["semantic_entailment_verified"] = True
    p = tmp_path / "live.json"
    p.write_text(json.dumps(obj), encoding="utf8")
    with pytest.raises(AgentEvaluationError):
        load_recorded_live_snapshot(p, "2510.22344")


def test_offline_comparison_and_saving_reports(tmp_path):
    suite = load_agent_suite(SUITE)
    calls = []
    def runner(question, paper_id, k):
        calls.append((question, paper_id, k))
        return mock_preview(suite, k)
    report = run_agent_evaluation(suite, data_dir=tmp_path, k_values=(3, 5),
                                  runner=runner, recorded_live=REPLAY)
    assert [t[2] for t in calls] == [3, 5]
    assert report["run_count"] == 2
    assert report["llm_calls"] == 0
    assert report["independently_labeled_ground_truth"] is False
    assert report["semantic_entailment_verified"] is False
    assert report["recorded_live_observation"]["accepted_claims"] == 4
    json_path, md_path = save_agent_evaluation(report, tmp_path)
    data = json.loads(json_path.read_text(encoding="utf8"))
    assert len(data["per_run"]) == 2
    assert "NOT independently judged qrels" in md_path.read_text(encoding="utf8")
    assert "OPENAI_API_KEY" not in md_path.read_text(encoding="utf8")


def test_cli_agent_evaluate_offline_using_stub(monkeypatch, tmp_path, capsys):
    from papertrail import cli
    from papertrail.agent.evaluation import run_agent_evaluation as actual
    suite = load_agent_suite(SUITE)
    monkeypatch.setenv("PAPERTRAIL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "DO_NOT_USE_FAKE_KEY")
    def fake_eval(found_suite, *, data_dir, k_values, recorded_live):
        return actual(found_suite, data_dir=data_dir, k_values=k_values,
                      runner=lambda q, p, k: mock_preview(suite, k), recorded_live=recorded_live)
    monkeypatch.setattr(cli, "run_agent_evaluation", fake_eval)
    assert cli.main(["agent-evaluate", "--suite", str(SUITE), "--k-values", "3,5",
                     "--recorded-live", str(REPLAY)]) == 0
    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["phase"] == "4.1"
    assert result["llm_calls"] == 0
    assert result["run_count"] == 2
    assert "DO_NOT_USE_FAKE_KEY" not in output
    assert Path(result["report_json"]).is_file()

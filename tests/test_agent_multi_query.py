"""Offline Phase 4.2.1 multi-query fusion/paired-evaluation contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from papertrail.agent.evaluation import AgentEvaluationError, load_agent_suite
from papertrail.agent.multi_eval import evaluate_multi_preview, run_multi_query_evaluation, save_multi_query_evaluation
from papertrail.agent.multi_query import fuse_validated_queries
from papertrail.agent.planning import analyze_question, build_research_plan
from papertrail.agent.query_rewriting import focused_queries
from papertrail.agent.tool_execution import ExecutionError, ExecutionOptions
from papertrail import cli

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/agent_fairrag_log_anchors_v1.json"
PAPER = "2510.22344"


def hit(name, rank, page=18, text=None):
    return {"chunk_id": name, "page_number": page, "citation": f"https://arxiv.org/abs/{PAPER}#page={page}",
            "rank": rank, "rrf_score": 0.02, "text": text or "Fixture only - unreviewed.", "context": []}


def test_rrf_is_deterministic_deduped_capped_and_provenance_preserving():
    branches = {"baseline": [hit("A", 1), hit("common", 2)],
                "focused": [hit("common", 1), hit("B", 2)]}
    first = fuse_validated_queries(branches, limit=2)
    assert first == fuse_validated_queries(branches, limit=2)
    assert [x["chunk_id"] for x in first] == ["common", "A"]
    assert all(x["rank"] == i for i, x in enumerate(first, 1))
    assert first[0]["rrf_score"] > first[1]["rrf_score"]
    assert first[0]["citation"].endswith("#page=18")
    assert not any("source_query" in x for x in first)


def test_rrf_rejects_conflicting_source_payload_and_bad_budget():
    with pytest.raises(ValueError, match="Conflicting"):
        fuse_validated_queries({"baseline": [hit("same", 1)],
                               "focused": [hit("same", 1, text="tampered")]}, limit=2)
    with pytest.raises(ValueError):
        fuse_validated_queries({"baseline": [hit("x", 1)], "focused": []}, limit=0)
    with pytest.raises(ValueError):
        fuse_validated_queries({"baseline": [hit("x", 1)] * 4, "focused": []}, limit=3)
    with pytest.raises(ValueError):
        fuse_validated_queries({"baseline": []}, limit=2)


def fake_runner(question, paper_id, k, style):
    case = load_agent_suite(SUITE).cases[0]
    plan = build_research_plan(question, paper_id=paper_id)
    primary = [x.search_query for x in plan.steps if x.proposed_tool == "hybrid_retrieval"]
    focused = focused_queries(question, analyze_question(question).intents)
    evidence = []
    selected = []
    tasks = []
    events = []
    for idx, facet in enumerate(case.facets):
        anchors = facet.anchor_chunk_ids
        # Synthetic diagnostic: baseline comparison, focused limitations,
        # both metrics. All other unlisted passages remain unjudged.
        if idx == 0:
            a = [anchors[0], "common_0", "bother_0"]
            b = ["common_0", "fother_0", "noise_0"]
        elif idx == 1:
            a = [anchors[0], "common_1", "bother_1"]
            b = [anchors[0], "fother_1", "noise_1"]
        else:
            a = ["common_2", "bother_2", "bother_3"]
            b = [anchors[0], anchors[1], "fother_2"]
        branches = {"baseline": a[:k], "focused": b[:k]}
        if style == "multi":
            out = fuse_validated_queries({key: [hit(ch, i) for i, ch in enumerate(ids, 1)]
                                          for key, ids in branches.items()}, limit=k)
            queries = [("baseline", primary[idx]), ("focused", focused[idx])]
            query = primary[idx]
        else:
            out = [hit(ch, i) for i, ch in enumerate(branches[style], 1)]
            queries = [(style, primary[idx] if style == "baseline" else focused[idx])]
            query = queries[0][1]
        evidence.append({"step_id": facet.step_id, "query": query, "hits": out,
                         "query_traces": [{"style": label, "query": q, "chunk_ids": branches[label]}
                                          for label, q in queries]})
        tasks.append({"id": facet.step_id, "search_query": query,
                      "proposed_tool": "hybrid_retrieval"})
        for label, q in queries:
            events.append({"name": "hybrid_retrieval", "status": "success",
                           "detail": {"step_id": facet.step_id, "style": label, "hits": len(branches[label])}})
        selected += [{"step_id": facet.step_id, "chunk_id": item["chunk_id"]} for item in out[:4]]
    return {"paper_id": paper_id, "query_style": style,
            "status": "ready_for_llm_synthesis", "llm_calls": 0, "network_opt_in": False,
            "generation_mode": "preview", "answer_generated": False,
            "evidence": evidence, "tool_events": events,
            "synthesis": {"source_register": selected}, "plan": tasks}


def test_three_way_comparison_uses_actual_post_budget_not_oracle(tmp_path):
    suite = load_agent_suite(SUITE)
    calls = []

    def runner(q, p, k, style):
        calls.append((k, style))
        return fake_runner(q, p, k, style)

    report = run_multi_query_evaluation(suite, data_dir=tmp_path, k_values=(3, 5), runner=runner)
    assert calls == [(3, "baseline"), (3, "focused"), (3, "multi"),
                     (5, "baseline"), (5, "focused"), (5, "multi")]
    assert report["run_count"] == 6
    assert report["llm_calls"] == 0 and report["network_opt_in"] is False
    assert report["default_agent_query_style_changed"] is False
    assert report["comparisons"][0]["union_anchors_pre_budget_theoretical"] == 4
    assert report["comparisons"][0]["multi_anchors_post_budget"] <= 4
    assert report["comparisons"][0]["retrieval_query_counts"]["multi"] == 6
    assert report["comparisons"][0]["retrieval_query_counts"]["baseline"] == 3
    assert "NOT gold" in report["interpretation"]
    assert report["comparisons"][0]["multi_fraction"] <= report["comparisons"][0]["union_fraction_theoretical"]
    jpath, mpath = save_multi_query_evaluation(report, tmp_path)
    assert json.loads(jpath.read_text(encoding="utf8"))["phase"] == "4.2.1"
    assert "Union (pre-budget, theoretical)" in mpath.read_text(encoding="utf8")
    assert "First observed anchor rank" in mpath.read_text(encoding="utf8")


def test_three_way_rejects_malformed_query_provenance():
    case = load_agent_suite(SUITE).cases[0]
    result = fake_runner(case.question, case.paper_id, 3, "multi")
    assert evaluate_multi_preview(result, case, 3)["query_strategy_audited"]
    result["evidence"][0]["query_traces"][0]["query"] = "altered"
    with pytest.raises(AgentEvaluationError, match="trace malformed"):
        evaluate_multi_preview(result, case, 3)
    result = fake_runner(case.question, case.paper_id, 3, "multi")
    result["evidence"][0]["hits"][0]["chunk_id"] = "fabricated"
    with pytest.raises(AgentEvaluationError, match="source subset"):
        evaluate_multi_preview(result, case, 3)


def test_guard_and_cli_default_unchanged(monkeypatch, capsys):
    assert ExecutionOptions().query_style == "baseline"
    with pytest.raises(ExecutionError, match="local-only"):
        ExecutionOptions(query_style="multi", allow_network=True)
    args = cli.build_parser().parse_args(["agent-answer", "--paper-id", PAPER,
                                           "--question", load_agent_suite(SUITE).cases[0].question])
    assert args.query_style == "baseline"
    assert cli.build_parser().parse_args(["agent-multi-query-evaluate"]).k_values == "3,5"
    assert cli.main(["agent-answer", "--paper-id", PAPER, "--question", args.question,
                     "--mode", "live", "--query-style", "multi"]) == 2
    assert "preview-only" in capsys.readouterr().err
    with pytest.raises(AgentEvaluationError):
        run_multi_query_evaluation(load_agent_suite(SUITE), data_dir=Path("data"),
                                   k_values=(5, 3), runner=fake_runner)

"""Phase 4.2.2 deterministic provenance, budget, and evaluation regression tests."""
from pathlib import Path
import json

import pytest

from papertrail.agent.budget_eval import run_budget_evaluation, save_budget_evaluation
from papertrail.agent.multi_query import fuse_budgeted_queries, fuse_validated_queries
from papertrail.agent.evaluation import AgentEvaluationError, load_agent_suite
from papertrail.agent.planning import build_research_plan, analyze_question
from papertrail.agent.query_rewriting import focused_queries
from papertrail.agent.tool_execution import ExecutionOptions, ExecutionError
from papertrail import cli

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/agent_fairrag_log_anchors_v1.json"
PAPER = "2510.22344"


def hit(cid, rank, text="Synthetic unreviewed passage", page=18):
    return {"chunk_id": cid, "rank": rank, "page_number": page,
            "citation": f"https://arxiv.org/abs/{PAPER}#page={page}",
            "text": text, "context": [], "rrf_score": 0.02}


def test_budgeted_interleaves_deduplicates_and_respects_limit():
    inputs = {"baseline": [hit("base_best", 1), hit("shared", 2), hit("base_only", 3)],
              "focused": [hit("focus_best", 1), hit("focus_second", 2), hit("shared", 3)]}
    rrf = fuse_validated_queries(inputs, limit=3)
    budget = fuse_budgeted_queries(inputs, limit=3)
    assert [p["chunk_id"] for p in budget] == ["focus_best", "base_best", "focus_second"]
    assert budget == fuse_budgeted_queries(inputs, limit=3)
    assert len(budget) <= 3
    assert all(p["rank"] == i for i, p in enumerate(budget, 1))
    assert all(p["rrf_score"] > 0 for p in budget)
    assert {p["chunk_id"] for p in budget}.issubset({p["chunk_id"] for l in inputs.values() for p in l})
    assert set(x["chunk_id"] for x in rrf) != set()  # old algorithm stays available


def test_budgeted_detects_provenance_conflict_and_overbudget():
    with pytest.raises(ValueError, match="Conflicting source provenance"):
        fuse_budgeted_queries({"baseline": [hit("same", 1)],
                              "focused": [hit("same", 1, text="Altered source")]}, limit=2)
    with pytest.raises(ValueError, match="Invalid multi-query budget"):
        fuse_budgeted_queries({"baseline": [], "focused": []}, limit=0)
    with pytest.raises(ValueError, match="Unbounded multi-query"):
        fuse_budgeted_queries({"baseline": [hit(str(i), i) for i in range(1, 4)],
                              "focused": []}, limit=2)
    assert fuse_budgeted_queries({"baseline": [hit("only", 1)], "focused": []}, limit=3)[0]["chunk_id"] == "only"
    assert fuse_budgeted_queries({"baseline": [], "focused": []}, limit=3) == []


def fake_runner(question, paper_id, k, style):
    case = load_agent_suite(SUITE).cases[0]
    base_queries = [x.search_query for x in build_research_plan(question, paper_id=paper_id).steps
                    if x.proposed_tool == "hybrid_retrieval"]
    focus_queries = focused_queries(question, analyze_question(question, paper_id=paper_id).intents)
    evidence, selected, tasks, events = [], [], [], []
    for index, facet in enumerate(case.facets):
        ids = facet.anchor_chunk_ids
        if index == 0:
            base, focus = [ids[0], "common0", "noise0"], ["common0", "fx0", "fn0"]
        elif index == 1:
            base, focus = [ids[0], "common1", "bn1"], [ids[0], "fn1", "fx1"]
        else:
            base, focus = ["noise2", "second2", "third2"], [ids[0], ids[1], "special2"]
        branches = {"baseline": [hit(cid, n) for n, cid in enumerate(base[:k], 1)],
                    "focused": [hit(cid, n) for n, cid in enumerate(focus[:k], 1)]}
        if style in {"budgeted", "multi"}:
            fuse = fuse_budgeted_queries if style == "budgeted" else fuse_validated_queries
            outputs = fuse(branches, limit=k)
            queried = (("baseline", base_queries[index]), ("focused", focus_queries[index]))
            query = base_queries[index]
        else:
            outputs = branches[style]
            queried = ((style, base_queries[index] if style == "baseline" else focus_queries[index]),)
            query = queried[0][1]
        traces = [{"style": name, "query": q, "chunk_ids": [hit["chunk_id"] for hit in branches[name]]}
                  for name, q in queried]
        evidence.append({"step_id": facet.step_id, "query": query, "hits": outputs, "query_traces": traces})
        tasks.append({"id": facet.step_id, "search_query": query, "proposed_tool": "hybrid_retrieval"})
        for name, q in queried:
            events.append({"name": "hybrid_retrieval", "status": "success",
                           "detail": {"step_id": facet.step_id, "style": name, "hits": len(branches[name])}})
        selected.extend({"step_id": facet.step_id, "chunk_id": h["chunk_id"]} for h in outputs[:4])
    return {"paper_id": paper_id, "query_style": style, "status": "ready_for_llm_synthesis",
            "llm_calls": 0, "network_opt_in": False, "generation_mode": "preview",
            "answer_generated": False, "evidence": evidence, "tool_events": events,
            "synthesis": {"source_register": selected}, "plan": tasks}


def test_four_way_budget_comparison_records_actual_hits_not_union(tmp_path):
    suite = load_agent_suite(SUITE)
    calls = []
    def runner(q, p, k, style):
        calls.append((k, style))
        return fake_runner(q, p, k, style)
    report = run_budget_evaluation(suite, data_dir=tmp_path, k_values=(3, 5), runner=runner)
    assert report["run_count"] == 8 and report["llm_calls"] == 0
    assert calls == [(k, style) for k in (3, 5) for style in ("baseline", "focused", "multi", "budgeted")]
    assert report["default_agent_query_style_changed"] is False
    for row in report["comparisons"]:
        assert row["union_anchors_pre_budget_theoretical"] == 4
        assert row["budgeted_anchors_post_budget"] <= row["union_anchors_pre_budget_theoretical"]
        assert row["retrieval_query_counts"] == {"baseline": 3, "focused": 3, "multi": 6, "budgeted": 6}
        assert row["budgeted_anchors_model_visible"] <= row["budgeted_anchors_post_budget"]
    jpath, mpath = save_budget_evaluation(report, tmp_path)
    assert json.loads(jpath.read_text(encoding="utf-8"))["run_count"] == 8
    assert "NOT gold" in report["interpretation"]
    assert "post-budget" in mpath.read_text(encoding="utf-8")


def test_paired_comparison_rejects_divergent_branches():
    suite = load_agent_suite(SUITE)
    def runner(q, p, k, style):
        result = fake_runner(q, p, k, style)
        if style == "budgeted":
            result["evidence"][0]["query_traces"][0]["chunk_ids"].reverse()
        return result
    with pytest.raises(AgentEvaluationError, match="Paired branch results differ"):
        run_budget_evaluation(suite, data_dir=Path("data"), k_values=(3,), runner=runner)


def test_budgeted_cli_safety_and_default(capsys):
    assert ExecutionOptions().query_style == "baseline"
    assert ExecutionOptions(query_style="budgeted", top_k=3).query_style == "budgeted"
    with pytest.raises(ExecutionError, match="local-only"):
        ExecutionOptions(query_style="budgeted", allow_network=True)
    assert cli.build_parser().parse_args(["agent-budget-evaluate"]).k_values == "3,5"
    assert cli.build_parser().parse_args(["agent-answer", "--question", "Sample question with enough words"]).query_style == "baseline"
    status = cli.main(["agent-answer", "--paper-id", PAPER,
                       "--question", load_agent_suite(SUITE).cases[0].question,
                       "--mode", "live", "--query-style", "budgeted"])
    assert status == 2
    assert "preview-only" in capsys.readouterr().err

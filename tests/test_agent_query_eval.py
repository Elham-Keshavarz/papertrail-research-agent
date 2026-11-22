"""Phase 4.2 paired, provisional, zero-API query interventions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from papertrail.agent.evaluation import AgentEvaluationError, load_agent_suite
from papertrail.agent.planning import analyze_question, build_research_plan
from papertrail.agent.query_eval import (evaluate_query_preview, run_query_evaluation,
                                         save_query_evaluation)
from papertrail.agent.query_rewriting import focused_queries
from papertrail.agent.tool_execution import ExecutionError, ExecutionOptions
from papertrail.agent.workflow import plan_research_node
from papertrail import cli

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/agent_fairrag_log_anchors_v1.json"
QUESTION = "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations."


def fake_runner(question, paper_id, k, style):
    suite = load_agent_suite(SUITE)
    case = suite.cases[0]
    plan = build_research_plan(question, paper_id=paper_id)
    intents = analyze_question(question, paper_id=paper_id).intents
    queries = focused_queries(question, intents) if style == "focused" else plan.analysis.subquestions
    evidence = []
    source_register = []
    tasks = []
    for idx, (facet, query) in enumerate(zip(case.facets, queries, strict=True)):
        hits = []
        # The focused strategy recovers the two missing limitations anchors,
        # this is a synthetic regression, NOT an observed FAIR-RAG score.
        ids = facet.anchor_chunk_ids if (idx < 2 or style == "focused") else []
        for item in ids:
            hits.append({"chunk_id": item})
        hits += [{"chunk_id": f"unjudged_{idx}_{j}"} for j in range(k)]
        hits = hits[:k]
        evidence.append({"step_id": facet.step_id, "query": query, "hits": hits})
        tasks.append({"id": facet.step_id, "search_query": query, "proposed_tool": "hybrid_retrieval"})
        for hit in hits[: min(k, 4)]:
            source_register.append({"step_id": facet.step_id, "chunk_id": hit["chunk_id"]})
    return {"paper_id": paper_id, "query_style": style, "status": "ready_for_llm_synthesis",
            "llm_calls": 0, "network_opt_in": False,
            "generation_mode": "preview", "answer_generated": False,
            "evidence": evidence, "synthesis": {"source_register": source_register},
            "plan": tasks}


def test_focused_queries_are_deterministic_and_not_hardcoded_paper_claims():
    analysis = analyze_question(QUESTION)
    assert analysis.intents == ["comparison", "metrics", "limitations"]
    queries = focused_queries(QUESTION, analysis.intents)
    assert "FAIR-RAG" in queries[2]
    assert "component-level failures" in queries[2]
    assert "architectural failures" in queries[2]
    assert "63.5" not in " ".join(queries)
    assert queries == focused_queries(QUESTION, analysis.intents)
    assert all(20 < len(q) < 500 for q in queries)


def test_generic_question_has_nonempty_topic_and_no_sourced_assertions():
    query = focused_queries("What architectural limitations did the authors describe?", ["limitations"])[0]
    assert "architectural" in query
    assert "limitations" in query
    assert "FAIR-RAG" not in query


@pytest.mark.parametrize("bad,intents", [(None, ["general"]), ("short", ["general"]),
                                             (QUESTION, []), (QUESTION, ["unsupported"]),
                                             (QUESTION, ["general"] * 6)])
def test_bad_focused_queries_rejected(bad, intents):
    with pytest.raises(ValueError):
        focused_queries(bad, intents)


def test_default_planner_query_unchanged_and_focused_only_changes_search_terms():
    base = plan_research_node({"question": QUESTION, "planner_requested": "deterministic"})
    custom = plan_research_node({"question": QUESTION, "planner_requested": "deterministic"},
                                query_style="focused")
    assert base["query_style"] == "baseline"
    assert custom["query_style"] == "focused"
    assert base["question_analysis"] == custom["question_analysis"]
    assert base["plan_validation"] == custom["plan_validation"]
    assert [x["search_query"] for x in base["plan"] if x["proposed_tool"] == "hybrid_retrieval"] != [
        x["search_query"] for x in custom["plan"] if x["proposed_tool"] == "hybrid_retrieval"]
    assert all(x["depends_on"] == y["depends_on"] for x, y in zip(base["plan"], custom["plan"], strict=True))
    assert len(custom["plan"]) == len(base["plan"])
    assert ExecutionOptions().query_style == "baseline"


def test_focused_rejects_llm_planner_and_invalid_options():
    with pytest.raises(ValueError):
        plan_research_node({"question": QUESTION, "planner_requested": "llm"}, query_style="focused")
    with pytest.raises(ExecutionError):
        ExecutionOptions(query_style="na")


def test_offline_pair_separates_observed_anchors_by_facet(tmp_path):
    suite = load_agent_suite(SUITE)
    calls = []

    def fake(question, paper_id, k, style):
        calls.append((paper_id, k, style))
        return fake_runner(question, paper_id, k, style)

    report = run_query_evaluation(suite, data_dir=tmp_path, k_values=(3, 5), runner=fake)
    assert report["run_count"] == 4
    assert report["llm_calls"] == 0 and report["network_opt_in"] is False
    assert not report["independently_labeled_ground_truth"]
    assert not report["default_agent_query_style_changed"]
    assert calls == [("2510.22344", 3, "baseline"), ("2510.22344", 3, "focused"),
                     ("2510.22344", 5, "baseline"), ("2510.22344", 5, "focused")]
    assert report["comparisons"][0]["delta_observed_anchor_fraction"] == 0.5
    assert report["comparisons"][0]["baseline_observed_anchor_fraction"] == 0.5
    assert report["comparisons"][0]["focused_observed_anchor_fraction"] == 1.0
    assert report["per_run"][0]["facets"][2]["first_observed_anchor_rank"] is None
    assert report["per_run"][1]["facets"][2]["first_observed_anchor_rank"] == 1
    j, m = save_query_evaluation(report, tmp_path)
    assert j.is_file() and m.is_file()
    assert json.loads(j.read_text())["phase"] == "4.2"
    assert "NOT scientific" in m.read_text()
    assert "component-level failures" in m.read_text()


def test_failed_preview_and_style_mismatch_fail_closed():
    suite = load_agent_suite(SUITE)
    case = suite.cases[0]
    item = fake_runner(case.question, case.paper_id, 3, "focused")
    item["llm_calls"] = 1
    with pytest.raises(AgentEvaluationError, match="Offline evaluation invariant"):
        evaluate_query_preview(item, case, 3, "focused")
    item["llm_calls"] = 0
    item["network_opt_in"] = True
    with pytest.raises(AgentEvaluationError):
        evaluate_query_preview(item, case, 3, "focused")
    item["network_opt_in"] = False
    with pytest.raises(AgentEvaluationError, match="Query strategy mismatch"):
        evaluate_query_preview(item, case, 3, "baseline")
    item["plan"][2]["search_query"] = "unexpected"
    with pytest.raises(AgentEvaluationError, match="disagrees"):
        evaluate_query_preview(item, case, 3, "focused")


def test_cli_parser_default_and_opt_in_guard(monkeypatch, capsys):
    args = cli.build_parser().parse_args(["agent-answer", "--paper-id", "2510.22344",
                                           "--question", QUESTION])
    assert args.query_style == "baseline"
    assert cli.main(["agent-answer", "--paper-id", "2510.22344", "--question", QUESTION,
                     "--mode", "live", "--query-style", "focused"]) == 2
    assert "preview-only" in capsys.readouterr().err
    eval_args = cli.build_parser().parse_args(["agent-query-evaluate", "--k-values", "3,5"])
    assert eval_args.command == "agent-query-evaluate"


def test_bad_k_not_accepted():
    with pytest.raises(AgentEvaluationError):
        run_query_evaluation(load_agent_suite(SUITE), data_dir=Path("data"), k_values=(5, 3), runner=fake_runner)

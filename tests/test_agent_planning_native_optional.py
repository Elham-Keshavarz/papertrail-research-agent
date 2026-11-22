"""Real graph 3.2 smoke tests (run on CI and Windows with agent extra)."""

import pytest
pytest.importorskip("langgraph", reason="Install agent extra for native planning workflow tests")

from papertrail.agent.workflow import run_agent_preview, build_research_graph
from papertrail.agent.planning import ResearchPlan


def test_native_research_plan_multiple_facets_and_paper_ready(tmp_path):
    paper_id = "2510.22344"
    for parent, filename in [
        ("chunks", paper_id + ".json"), ("embeddings", paper_id + ".npy"),
        ("embeddings", paper_id + ".manifest.json"), ("indexes", paper_id + ".faiss"),
        ("indexes", paper_id + ".manifest.json"),
    ]:
        path = tmp_path / parent / filename
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(b"NOT INTEGRITY CHECKED")
    result = run_agent_preview(
        "Compare FAIR-RAG and baseline RAG on datasets and evaluation metrics.",
        paper_id=paper_id, data_dir=tmp_path,
    )
    assert result["status"] == "local_artifacts_present_unverified"
    assert result["plan_validation"] == "validated_dependency_dag"
    assert result["question_analysis"]["requires_comparison"] is True
    assert len([p for p in result["plan"] if p["proposed_tool"] == "hybrid_retrieval"]) >= 3
    assert result["llm_calls"] == result["tool_calls"] == 0
    assert not result["research_executed"] and not result["answer_generated"]
    assert [event["node"] for event in result["trace"]] == [
        "validate_request", "plan_research", "inspect_local_artifacts", "ready", "finalize",
    ]


def test_native_llm_requested_without_secret_deterministically_falls_back(tmp_path):
    result = run_agent_preview(
        "What evidence supports these benchmark evaluation metrics?", data_dir=tmp_path,
        planner_mode="llm", planner_api_key=None,
    )
    assert result["planner"]["requested"] == "llm"
    assert result["planner"]["used"] == "deterministic"
    assert result["planner"]["fallback_reason"] == "provider_not_configured"
    assert result["llm_calls"] == 0
    assert result["status"] == "needs_paper_selection"


def test_native_provider_contributes_questions_without_tool_execution(tmp_path):
    from papertrail.agent.workflow import plan_research_node
    provider = lambda *_: ["Which source passages explain the evaluation datasets?"]
    graph = build_research_graph(provider)
    result = graph.invoke({
        "question": "Which datasets evaluate this method?", "paper_id": None,
        "max_papers": 5, "data_dir": str(tmp_path),
        "planner_requested": "llm", "trace": [],
    }, config={"recursion_limit": 16})
    assert result["planner_used"] == "llm_proposal_unverified"
    assert result["llm_calls"] == 1
    assert result["plan"][3]["search_query"] == "Which source passages explain the evaluation datasets?"
    assert result["status"] == "needs_paper_selection"

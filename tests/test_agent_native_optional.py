"""Real LangGraph smoke tests: run in CI and user's Windows environment."""

import pytest

pytest.importorskip("langgraph", reason="Install the agent extra for native StateGraph tests")

from papertrail.agent import build_research_graph, run_agent_preview


def test_real_graph_without_paper_uses_branch(tmp_path):
    result = run_agent_preview("Which datasets evaluate the research approach?", data_dir=tmp_path)
    assert result["status"] == "needs_paper_selection"
    assert [event["node"] for event in result["trace"]] == [
        "validate_request", "plan_research", "inspect_local_artifacts", "needs_paper", "finalize",
    ]
    assert result["answer_generated"] is False


def test_real_graph_missing_assets_branch(tmp_path):
    result = run_agent_preview("What are the limitations of this method?", paper_id="2510.22344", data_dir=tmp_path)
    assert result["status"] == "missing_local_artifacts"
    assert result["artifact_check"]["missing"]
    assert result["tool_calls"] == 0


def test_real_graph_ready_branch_and_state_isolation(tmp_path):
    paper = "2510.22344"
    for folder, filename in [
        ("chunks", paper + ".json"),
        ("embeddings", paper + ".npy"),
        ("embeddings", paper + ".manifest.json"),
        ("indexes", paper + ".faiss"),
        ("indexes", paper + ".manifest.json"),
    ]:
        path = tmp_path / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    a = run_agent_preview("Why does this model use multiple iterations?", paper_id=paper, data_dir=tmp_path)
    b = run_agent_preview("How is its evaluation performed?", paper_id=paper, data_dir=tmp_path)
    assert a["status"] == b["status"] == "local_artifacts_present_unverified"
    assert len(a["trace"]) == len(b["trace"]) == 5
    assert a["artifact_check"]["check_type"] == "existence_only_not_integrity_verified"


def test_real_graph_invalid_request_routes_without_crash():
    result = build_research_graph().invoke({"question": "no", "paper_id": None, "trace": []})
    assert result["status"] == "invalid_request"
    assert [e["node"] for e in result["trace"]] == ["validate_request", "finalize"]
    assert "plan" not in result

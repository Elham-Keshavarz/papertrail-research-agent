"""Pure offline tests for the Phase 3.1 graph state, routing, and safety."""

from pathlib import Path
import sys
import types

import pytest

from papertrail.agent.workflow import (
    ARTIFACT_LABELS,
    build_research_graph,
    finalize_node,
    inspect_local_artifacts_node,
    missing_node,
    needs_paper_node,
    plan_research_node,
    ready_node,
    route_after_inspection,
    route_after_validation,
    validate_request_node,
    run_agent_preview,
)


def test_valid_request_normalized():
    result = validate_request_node({"question": "  How    does    retrieval reduce hallucination?   ", "max_papers": 2})
    assert result["question"] == "How does retrieval reduce hallucination?"
    assert result["paper_id"] is None
    assert result["status"] == "validated"
    assert route_after_validation(result) == "plan"


@pytest.mark.parametrize("question", ["too short", "", "x" * 1001, None])
def test_invalid_question_routes_to_finalizer(question):
    result = validate_request_node({"question": question})
    assert result["status"] == "invalid_request"
    assert result["errors"]
    assert route_after_validation(result) == "finalize"


def test_paper_id_traversal_rejected():
    result = validate_request_node({"question": "How many authors wrote this paper?", "paper_id": "../../etc/passwd"})
    assert result["status"] == "invalid_request"
    assert route_after_validation(result) == "finalize"


def test_invalid_max_papers_rejected():
    assert validate_request_node({"question": "What is the impact of retrieval?", "max_papers": 100})["status"] == "invalid_request"


def test_plan_has_expected_unexecuted_steps():
    plan = plan_research_node({"question": "What is the evidence quality?"})
    assert [p["id"] for p in plan["plan"]] == ["discover", "ingest", "prepare", "retrieve_01", "retrieve_02", "assess", "answer"]
    assert all(p["execution"] == "not_executed" for p in plan["plan"])
    assert plan == plan_research_node({"question": "What is the evidence quality?"})


def test_no_paper_selected_yields_explicit_status(tmp_path):
    inspection = inspect_local_artifacts_node({"paper_id": None, "data_dir": str(tmp_path)})
    assert route_after_inspection({**inspection, "paper_id": None}) == "needs_paper"
    assert inspection["artifact_check"]["missing"] == list(ARTIFACT_LABELS)
    assert needs_paper_node({})["status"] == "needs_paper_selection"


def test_missing_local_assets_are_identified(tmp_path):
    inspection = inspect_local_artifacts_node({"paper_id": "2510.22344", "data_dir": str(tmp_path)})
    assert inspection["artifact_check"]["present"] == []
    assert inspection["artifact_check"]["ready"] is False
    assert route_after_inspection({**inspection, "paper_id": "2510.22344"}) == "missing"
    assert missing_node({})["status"] == "missing_local_artifacts"


def test_partial_local_assets_are_not_marked_ready(tmp_path):
    chunks = tmp_path / "chunks" / "2510.22344.json"
    chunks.parent.mkdir(parents=True)
    chunks.write_text("not parsed", encoding="utf-8")
    inspected = inspect_local_artifacts_node({"paper_id": "2510.22344", "data_dir": str(tmp_path)})["artifact_check"]
    assert inspected["present"] == ["chunks"]
    assert not inspected["ready"]
    assert len(inspected["missing"]) == 4


def test_all_files_present_only_claims_existence(tmp_path):
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
        path.write_bytes(b"not validated")
    check = inspect_local_artifacts_node({"paper_id": paper, "data_dir": str(tmp_path)})["artifact_check"]
    assert check["ready"] is True
    assert check["present"] == list(ARTIFACT_LABELS)
    assert check["check_type"] == "existence_only_not_integrity_verified"
    assert route_after_inspection({"paper_id": paper, "artifact_check": check}) == "ready"
    assert ready_node({})["status"] == "local_artifacts_present_unverified"


def test_defense_in_depth_rejects_invalid_id(tmp_path):
    with pytest.raises(ValueError, match="Invalid arXiv"):
        inspect_local_artifacts_node({"paper_id": "../outside", "data_dir": str(tmp_path)})


def test_finalize_never_claims_answer():
    assert finalize_node({"status": "validated"}) == {"trace": [{"node": "finalize", "event": "no_answer_generated"}]}


def test_graph_builder_wires_exact_native_api(monkeypatch):
    """Contract check without installing langgraph, validating graph topology."""
    recorded = {}

    class FakeGraph:
        def __init__(self, schema):
            recorded["schema"] = schema
            recorded["nodes"] = {}
            recorded["edges"] = []
            recorded["conditional"] = []

        def add_node(self, name, fn):
            recorded["nodes"][name] = fn

        def add_edge(self, source, target):
            recorded["edges"].append((source, target))

        def add_conditional_edges(self, source, fn, mapping):
            recorded["conditional"].append((source, fn, mapping))

        def compile(self):
            return self

    fake = types.ModuleType("langgraph")
    fake_graph_module = types.ModuleType("langgraph.graph")
    fake_graph_module.StateGraph = FakeGraph
    fake_graph_module.START = "__start__"
    fake_graph_module.END = "__end__"
    monkeypatch.setitem(sys.modules, "langgraph", fake)
    monkeypatch.setitem(sys.modules, "langgraph.graph", fake_graph_module)
    assert build_research_graph().__class__ is FakeGraph
    assert set(recorded["nodes"]) == {"validate_request", "plan_research", "inspect_local_artifacts", "ready", "needs_paper", "missing", "finalize"}
    assert ("__start__", "validate_request") in recorded["edges"]
    assert ("finalize", "__end__") in recorded["edges"]
    assert recorded["conditional"][0][2] == {"plan": "plan_research", "finalize": "finalize"}
    assert recorded["conditional"][1][2] == {"ready": "ready", "needs_paper": "needs_paper", "missing": "missing"}


def test_run_agent_preview_reports_zero_llm_and_tools(monkeypatch, tmp_path):
    """Mock graph.invoke only; all real node logic has separate tests."""
    import papertrail.agent.workflow as wf

    class FakeRunner:
        def invoke(self, payload, config):
            assert payload["data_dir"] == str(tmp_path)
            assert config["recursion_limit"] == 16
            return {
                "status": "needs_paper_selection", "paper_id": None,
                "plan": plan_research_node(payload)["plan"],
                "trace": [{"node": "finalize", "event": "no_answer_generated"}],
            }

    monkeypatch.setattr(wf, "build_research_graph", lambda: FakeRunner())
    result = run_agent_preview("Which methods are discussed here?", data_dir=tmp_path)
    assert result["status"] == "needs_paper_selection"
    assert result["tool_calls"] == result["llm_calls"] == 0
    assert result["research_executed"] is result["answer_generated"] is False
    assert len(result["plan"]) == 6


def test_run_agent_preview_rejects_short_question_before_invocation():
    with pytest.raises(Exception):
        run_agent_preview("too short")


def test_run_agent_preview_rejects_unsafe_id_before_invocation():
    with pytest.raises(ValueError, match="Invalid arXiv"):
        run_agent_preview("What makes this model reliable?", paper_id="../../bad")

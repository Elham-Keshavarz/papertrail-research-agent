"""Phase 3.4: conservative offline assessment, bounded retries and fail-closed outputs."""
import json
import sys
import types

import pytest

from papertrail.agent.assessment import assess_evidence, refinement_query
from papertrail.agent.tool_execution import ExecutionOptions, _paths, run_agent_research

PAPER = "2510.22344"
QUESTION = "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations."
CITATION = f"https://arxiv.org/abs/{PAPER}#page=9"


def hit(chunk, page=9, rank=1):
    return {"chunk_id": chunk, "page_number": page, "rank": rank, "rrf_score": 0.03,
            "citation": f"https://arxiv.org/abs/{PAPER}#page={page}",
            "text": f"Simulated source text {chunk} without any verified scientific assertions", "context": []}


def plan(*names):
    return [{"id": name, "search_query": f"Search for {name}",
             "proposed_tool": "hybrid_retrieval", "execution": "not_executed"} for name in names]


def test_structural_assessment_never_claims_semantic_factual_sufficiency():
    report = assess_evidence(plan("a", "b"), [
        {"step_id": "a", "hits": [hit("a1", 9), hit("a2", 9)]},
        {"step_id": "b", "hits": [hit("b1", 13), hit("b2", 14)]},
    ])
    assert report["decision"] == "structurally_covered_unverified"
    assert report["subquestions_meeting_threshold"] == 2
    assert report["distinct_source_pages"] == [9, 13, 14]
    assert report["semantic_coverage_verified"] is False
    assert report["factually_sufficient"] is None
    assert report["contradiction_analysis_performed"] is False
    assert report["answer_generation_allowed"] is False
    assert any("one PDF page" in w for w in report["warnings"])


def test_missing_facet_not_hidden_by_hits_on_another_facet():
    report = assess_evidence(plan("a", "b"), [
        {"step_id": "a", "hits": [hit("a1"), hit("a2"), hit("a3")]},
        {"step_id": "b", "hits": []},
    ])
    assert report["decision"] == "needs_more_retrieval"
    assert report["gap_step_ids"] == ["b"]


def test_duplicate_chunk_ids_do_not_count_toward_threshold():
    report = assess_evidence(plan("a"), [{"step_id": "a", "hits": [hit("repeat"), hit("repeat")]}])
    assert report["gap_step_ids"] == ["a"]
    assert report["distinct_primary_chunks"] == 1


def test_cross_facet_duplicates_are_reported_but_not_claimed_as_verified():
    report = assess_evidence(plan("a", "b"), [
        {"step_id": "a", "hits": [hit("shared"), hit("a2")]},
        {"step_id": "b", "hits": [hit("shared"), hit("b2")]},
    ])
    assert report["cross_subquestion_duplicate_chunk_ids"] == ["shared"]
    assert report["decision"] == "structurally_covered_unverified"


def test_no_steps_is_not_sufficient():
    report = assess_evidence([], [])
    assert report["decision"] == "no_retrieval_steps"
    assert not report["citation_format_validated_by_retrieval"]


@pytest.mark.parametrize("n,k", [(0,3), (11,3), (2,0), (2,11)])
def test_threshold_rejects_invalid_values(n, k):
    with pytest.raises(ValueError):
        assess_evidence([], [], min_hits_per_step=n, top_k=k)


def test_threshold_clamps_to_requested_top_k():
    report = assess_evidence(plan("a"), [{"step_id": "a", "hits": [hit("only")]}],
                             min_hits_per_step=3, top_k=1)
    assert report["minimum_unique_chunks_per_step"] == 1
    assert report["decision"] == "structurally_covered_unverified"


def test_retry_query_only_uses_user_input():
    assert refinement_query(original_question="  Where   are sources? ",
                            planned_query="Find references. Topic: Where are sources?") == "Where are sources?"
    assert refinement_query(original_question="Where are sources?", planned_query="Where are sources?") is None
    assert refinement_query(original_question="   ", planned_query="Some query") is None
    assert len(refinement_query(original_question="x"*1000, planned_query="q")) == 500


class MiniGraph:
    def __init__(self, _state):
        self.nodes, self.edges, self.routes = {}, {}, {}

    def add_node(self, name, node):
        self.nodes[name] = node

    def add_edge(self, src, dst):
        self.edges[src] = dst

    def add_conditional_edges(self, src, route):
        self.routes[src] = route

    def compile(self):
        return self

    def invoke(self, state, config=None):
        state = dict(state)
        node, executions = self.edges["__start__"], 0
        while node != "__end__":
            executions += 1
            assert executions <= (config or {}).get("recursion_limit", 48)
            patch = self.nodes[node](state)
            for key, value in patch.items():
                state[key] = state.get(key, []) + value if key == "trace" else value
            node = self.routes[node](state) if node in self.routes else self.edges[node]
        return state


@pytest.fixture
def graph_shim(monkeypatch):
    graph = types.ModuleType("langgraph.graph")
    graph.StateGraph, graph.START, graph.END = MiniGraph, "__start__", "__end__"
    monkeypatch.setitem(sys.modules, "langgraph", types.ModuleType("langgraph"))
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph)


def seed_cache(path):
    paths = _paths(path, PAPER)
    for name in ("chunks", "vectors", "embedding_manifest"):
        paths[name].parent.mkdir(exist_ok=True, parents=True)
        paths[name].write_bytes(b"fixture: production adapter validates true hashes")


class FakeTools:
    def __init__(self, initial=1, retry=2, repeat=False, fail_retry=False, mismatch_retry=False):
        self.calls, self.queries = [], []
        self.initial, self.retry = initial, retry
        self.repeat, self.fail_retry, self.mismatch_retry = repeat, fail_retry, mismatch_retry

    def discover(self, *_):
        raise AssertionError("No network access requested")

    def ingest(self, *_):
        raise AssertionError("Should use cached paper")

    def chunk(self, *_):
        raise AssertionError("Should use cached chunks")

    def embed(self, *_):
        raise AssertionError("Should use cached vectors")

    def index(self, *_):
        self.calls.append("index")
        return {"cached": True, "vectors": 4}

    def retrieve(self, _paper_id, query, options):
        self.calls.append("retrieve")
        self.queries.append(query)
        retry = query == QUESTION
        if retry and self.fail_retry:
            raise RuntimeError("never-print-secret-key")
        if retry and self.mismatch_retry:
            forged = hit("forged")
            forged["citation"] = "https://arxiv.org/abs/9999.99999#page=9"
            return [forged]
        total = self.retry if retry else self.initial
        key_prefix = "original" if self.repeat or not retry else "retry"
        return [hit(f"{key_prefix}-{i}", rank=i+1) for i in range(total)]


def run(tmp_path, tool, **overrides):
    seed_cache(tmp_path)
    opts = ExecutionOptions(data_dir=tmp_path, **overrides)
    return run_agent_research(QUESTION, paper_id=PAPER, options=opts,
                              tools=tool, assess_mode=True)


def test_complete_evidence_assessed_without_extra_calls(graph_shim, tmp_path):
    fake = FakeTools(initial=3)
    result = run(tmp_path, fake)
    assert result["status"] == "evidence_assessed_unverified"
    assert result["tool_calls"] == 4
    assert len(fake.queries) == 3
    assert result["assessment"]["decision"] == "structurally_covered_unverified"
    assert result["plan"][-2]["execution"] == "completed_structural_unverified"
    assert result["plan"][-1]["execution"] == "not_executed"
    assert result["llm_calls"] == 0 and not result["answer_generated"]


def test_missing_chosen_facets_trigger_one_bounded_retry_each(graph_shim, tmp_path):
    fake = FakeTools(initial=1, retry=2)
    result = run(tmp_path, fake)
    assert result["status"] == "evidence_assessed_unverified"
    assert result["assessment"]["decision"] == "structurally_covered_unverified"
    assert result["tool_calls"] == 7
    assert fake.queries.count(QUESTION) == 3
    assert result["retry_attempts"] == {
        "retrieve_01": 1, "retrieve_02": 1, "retrieve_03": 1,
    }
    assert len(result["retry_justification"]) == 3
    assert result["assessment"]["distinct_primary_chunks"] == 3  # repeated across facets
    assert not result["answer_generated"]


def test_retries_that_find_no_novel_evidence_remain_gaps(graph_shim, tmp_path):
    fake = FakeTools(initial=1, retry=1, repeat=True)
    result = run(tmp_path, fake)
    assert result["status"] == "evidence_gaps_detected"
    assert len(result["assessment"]["gap_step_ids"]) == 3
    assert fake.calls == ["index"] + ["retrieve"] * 6
    assert result["assessment"]["answer_generation_allowed"] is False


def test_retry_may_be_disabled_without_extra_calls(graph_shim, tmp_path):
    fake = FakeTools(initial=1)
    result = run(tmp_path, fake, max_refinement_rounds=0)
    assert result["status"] == "evidence_gaps_detected"
    assert result["tool_calls"] == 4
    assert result["retry_attempts"] == {}


def test_budget_exhausted_does_not_execute_extra_retry(graph_shim, tmp_path):
    fake = FakeTools(initial=1)
    result = run(tmp_path, fake, max_tool_calls=4)
    assert result["status"] == "tool_budget_exhausted"
    assert result["tool_calls"] == 4
    assert len(fake.calls) == 4
    assert result["assessment"]["decision"] == "needs_more_retrieval"
    assert result["errors"]


def test_retry_tool_failure_redacts_exception_detail(graph_shim, tmp_path):
    fake = FakeTools(initial=1, fail_retry=True)
    result = run(tmp_path, fake)
    assert result["status"] == "tool_failed"
    assert "never-print-secret-key" not in json.dumps(result)
    assert result["tool_calls"] == 5


def test_retry_rejects_forged_citation(graph_shim, tmp_path):
    result = run(tmp_path, FakeTools(initial=1, mismatch_retry=True))
    assert result["status"] == "tool_failed"
    assert result["assessment"]["answer_generation_allowed"] is False
    assert all(h["citation"] != "https://arxiv.org/abs/9999.99999#page=9"
               for e in result["evidence"] for h in e["hits"])


def test_no_paper_does_not_try_retrieval_or_assess(graph_shim, tmp_path):
    fake = FakeTools()
    result = run_agent_research(QUESTION, options=ExecutionOptions(data_dir=tmp_path),
                                tools=fake, assess_mode=True)
    assert result["status"] == "needs_paper_selection"
    assert not result["research_executed"]
    assert result["assessment"] is None
    assert fake.calls == []


def test_bounded_configuration_rejects_too_many_retries():
    for kwargs in ({"max_refinement_rounds": 2}, {"max_refinement_rounds": -1},
                   {"min_hits_per_step": 0}, {"min_hits_per_step": 11}):
        with pytest.raises(ValueError):
            ExecutionOptions(**kwargs)


def test_retry_uses_only_user_original_question(graph_shim, tmp_path):
    fake = FakeTools(initial=1)
    result = run(tmp_path, fake)
    assert all(t["new_query"] == QUESTION for t in result["retry_justification"])
    assert not result["assessment"]["semantic_coverage_verified"]

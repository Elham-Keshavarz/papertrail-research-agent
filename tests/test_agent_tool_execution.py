"""No-network unit/contract tests for Phase 3.3. Graph shim tests do not replace native tests."""
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from papertrail.agent.tool_execution import (
    ExecutionError, ExecutionOptions, PaperTrailTools, _paths, _source_audit,
    build_tool_graph, run_agent_research,
)

PAPER = "2510.22344"
QUESTION = "Compare FAIR-RAG and traditional RAG on evaluation metrics and limitations."


class MiniGraph:
    """Test-only state machine for node/route behavior; native tests use actual LangGraph."""
    def __init__(self, state_type):
        self.nodes = {}
        self.edges = {}
        self.routes = {}

    def add_node(self, name, action):
        self.nodes[name] = action

    def add_edge(self, source, target):
        self.edges[source] = target

    def add_conditional_edges(self, name, route):
        self.routes[name] = route

    def compile(self):
        return self

    def invoke(self, initial, config=None):
        state = dict(initial)
        node = self.edges["__start__"]
        count = 0
        while node != "__end__":
            count += 1
            if count > (config or {}).get("recursion_limit", 32):
                raise RuntimeError("Graph recursion budget exceeded")
            update = self.nodes[node](state)
            for key, value in update.items():
                state[key] = state.get(key, []) + value if key == "trace" else value
            node = self.routes[node](state) if node in self.routes else self.edges[node]
        return state


@pytest.fixture
def shim_graph(monkeypatch):
    root = types.ModuleType("langgraph")
    sub = types.ModuleType("langgraph.graph")
    sub.StateGraph = MiniGraph
    sub.START = "__start__"
    sub.END = "__end__"
    monkeypatch.setitem(sys.modules, "langgraph", root)
    monkeypatch.setitem(sys.modules, "langgraph.graph", sub)


class FakeTools:
    def __init__(self, candidates=None, fail_on=None, invalid_citation=False):
        self.calls = []
        self.candidates = candidates or []
        self.fail_on = fail_on
        self.invalid_citation = invalid_citation

    def _record(self, name):
        self.calls.append(name)
        if name == self.fail_on:
            raise RuntimeError("secret-api-key=not-for-output")

    def discover(self, question, options):
        self._record("discover")
        return self.candidates

    def ingest(self, paper_id, options):
        self._record("ingest")
        return {"pages_extracted": 2}

    def chunk(self, paper_id, options):
        self._record("chunk")
        return {"chunks": 4}

    def embed(self, paper_id, options):
        self._record("embed")
        return {"vectors": 4}

    def index(self, paper_id, options):
        self._record("index")
        return {"cached": True, "vectors": 4}

    def retrieve(self, paper_id, query, options):
        self._record("retrieve")
        return [{"chunk_id": "c1", "page_number": 9,
                 "citation": "https://arxiv.org/abs/2510.22344#page=" +
                     ("12" if self.invalid_citation else "9"),
                 "rank": 1, "rrf_score": 0.03, "text": "Quoted PDF text",
                 "context": []}]


def seed_cache(folder, paper=PAPER):
    paths = _paths(folder, paper)
    for name in ("chunks", "vectors", "embedding_manifest"):
        path = paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dummy - verified by production adapter, not the fake")


def test_phase33_offline_reuses_artifacts_and_runs_tools(shim_graph, tmp_path):
    seed_cache(tmp_path)
    tool = FakeTools()
    result = run_agent_research(QUESTION, paper_id=PAPER,
                                options=ExecutionOptions(data_dir=tmp_path), tools=tool)
    assert result["status"] == "evidence_retrieved_no_answer"
    assert result["plan_validation"] == "validated_dependency_dag"
    assert result["tool_calls"] == 4
    assert tool.calls == ["index", "retrieve", "retrieve", "retrieve"]
    assert result["llm_calls"] == 0
    assert result["research_executed"] and not result["answer_generated"]
    assert len(result["evidence"]) == 3
    assert all(e["hits"][0]["citation"].endswith("#page=9") for e in result["evidence"])
    assert result["source_audit"]["pdf_extraction_quality_verified"] is False
    assert any(e["status"] == "reused_existing_artifact" for e in result["tool_events"])
    assert result["trace"][-1] == {"node": "finalize", "event": "no_answer_generated"}


def test_offline_without_id_never_discovers(shim_graph, tmp_path):
    tool = FakeTools()
    result = run_agent_research(QUESTION, options=ExecutionOptions(data_dir=tmp_path), tools=tool)
    assert result["status"] == "needs_paper_selection"
    assert result["tool_calls"] == 0
    assert tool.calls == []


def test_offline_missing_source_never_downloads(shim_graph, tmp_path):
    tool = FakeTools()
    result = run_agent_research(QUESTION, paper_id=PAPER,
                                options=ExecutionOptions(data_dir=tmp_path), tools=tool)
    assert result["status"] == "offline_source_missing"
    assert not tool.calls and result["tool_calls"] == 0
    assert not result["answer_generated"]


def test_discover_returns_candidates_without_auto_ingestion(shim_graph, tmp_path):
    tool = FakeTools(candidates=[{"paper_id": PAPER, "title": "FAIR-RAG", "url": "https://arxiv.org/abs/2510.22344"}])
    result = run_agent_research(QUESTION,
            options=ExecutionOptions(data_dir=tmp_path, allow_network=True), tools=tool)
    assert result["status"] == "paper_selection_required"
    assert result["discovery_candidates"][0]["paper_id"] == PAPER
    assert result["tool_calls"] == 1 and tool.calls == ["discover"]
    assert not result["answer_generated"]


def test_explicit_auto_select_first_and_no_more_network_with_cache(shim_graph, tmp_path):
    seed_cache(tmp_path)
    tool = FakeTools(candidates=[{"paper_id": PAPER, "title": "FAIR-RAG"}])
    result = run_agent_research(QUESTION,
         options=ExecutionOptions(data_dir=tmp_path, allow_network=True, auto_select_first=True),
         tools=tool)
    assert result["status"] == "evidence_retrieved_no_answer"
    assert result["paper_id"] == PAPER
    assert tool.calls == ["discover", "index", "retrieve", "retrieve", "retrieve"]


def test_budget_enforced_during_retrieval(shim_graph, tmp_path):
    seed_cache(tmp_path)
    tool = FakeTools()
    result = run_agent_research(QUESTION, paper_id=PAPER,
        options=ExecutionOptions(data_dir=tmp_path, max_tool_calls=2), tools=tool)
    assert result["status"] == "tool_budget_exhausted"
    assert result["tool_calls"] == 2
    assert len(result["evidence"]) == 1
    assert tool.calls == ["index", "retrieve"]


def test_failed_tool_keeps_error_type_but_never_exception_text(shim_graph, tmp_path):
    seed_cache(tmp_path)
    tool = FakeTools(fail_on="retrieve")
    result = run_agent_research(QUESTION, paper_id=PAPER,
                                options=ExecutionOptions(data_dir=tmp_path), tools=tool)
    assert result["status"] == "tool_failed"
    assert result["tool_calls"] == 2
    assert "RuntimeError" in result["errors"][0]
    assert "secret" not in json.dumps(result)


def test_reject_mismatched_citation_instead_of_emitting_fabricated_evidence(shim_graph, tmp_path):
    seed_cache(tmp_path)
    result = run_agent_research(QUESTION, paper_id=PAPER,
        options=ExecutionOptions(data_dir=tmp_path), tools=FakeTools(invalid_citation=True))
    assert result["status"] == "tool_failed"
    assert result["evidence"] == []


def test_no_candidates_and_invalid_candidate_fail_closed(shim_graph, tmp_path):
    empty = run_agent_research(QUESTION,
        options=ExecutionOptions(data_dir=tmp_path, allow_network=True), tools=FakeTools())
    assert empty["status"] == "no_papers_found"
    invalid = run_agent_research(QUESTION,
        options=ExecutionOptions(data_dir=tmp_path, allow_network=True),
        tools=FakeTools(candidates=[{"paper_id": "../../tmp", "title": "malicious"}]))
    assert invalid["status"] == "tool_failed"
    assert "../../tmp" not in json.dumps(invalid)


def test_planner_llm_requested_without_key_falls_back(shim_graph, tmp_path):
    seed_cache(tmp_path)
    result = run_agent_research(QUESTION, paper_id=PAPER,
        options=ExecutionOptions(data_dir=tmp_path), planner_mode="llm", tools=FakeTools())
    assert result["planner"]["fallback_reason"] == "provider_not_configured"
    assert result["llm_calls"] == 0


def test_invalid_budgets_and_flags_rejected():
    for kwargs in ({"max_tool_calls": 0}, {"top_k": True}, {"context_window": 9},
                   {"auto_select_first": True}, {"max_papers": 21}):
        with pytest.raises((ExecutionError, ValueError)):
            ExecutionOptions(**kwargs)
    for identifier in ("../../file", "bad id", "localhost"):
        with pytest.raises(ValueError):
            _paths(Path("data"), identifier)


def test_pdf_extracted_provenance_hash_match_and_mismatch(tmp_path):
    import pymupdf
    from papertrail.ingestion.pdf_extractor import extract_pdf
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Testing local source page 1")
    blob = doc.tobytes()
    doc.close()
    paths = _paths(tmp_path, PAPER)
    paths["pdf"].parent.mkdir(parents=True)
    paths["extracted"].parent.mkdir(parents=True)
    paths["pdf"].write_bytes(blob)
    report = extract_pdf(blob, PAPER)
    paths["extracted"].write_text(report.model_dump_json(), encoding="utf-8")
    assert _source_audit(PAPER, tmp_path)["status"] == "pdf_extracted_hash_matched"
    paths["pdf"].write_bytes(blob + b"tampered")
    with pytest.raises(ExecutionError, match="hashes do not match"):
        _source_audit(PAPER, tmp_path)


def test_adapter_discovery_no_network_protected(tmp_path):
    with pytest.raises(ExecutionError):
        PaperTrailTools().discover(QUESTION, ExecutionOptions(data_dir=tmp_path))


def test_adapter_offline_pdf_from_cache_never_fetches(tmp_path, monkeypatch):
    import pymupdf
    from papertrail.ingestion import pdf_downloader
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Testing real offline PDF ingestion")
    blob = doc.tobytes()
    doc.close()
    pdf_path = _paths(tmp_path, PAPER)["pdf"]
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(blob)
    def forbidden(self, paper_id):
        raise RuntimeError("Network fetch unexpectedly called")
    monkeypatch.setattr(pdf_downloader.ArxivPdfDownloader, "fetch", forbidden)
    result = PaperTrailTools().ingest(PAPER, ExecutionOptions(data_dir=tmp_path))
    assert result["cached_pdf"] is True
    assert result["pages_extracted"] == 1
    assert _paths(tmp_path, PAPER)["extracted"].is_file()


def test_graph_backend_mandatory_for_actual_execution(monkeypatch):
    import builtins
    original = builtins.__import__
    def no_langgraph(name, *args, **kwargs):
        if name == "langgraph.graph":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_langgraph)
    with pytest.raises(Exception, match="LangGraph missing"):
        build_tool_graph(ExecutionOptions())


def test_offline_guard_prevents_hf_model_resolution_and_restores_env(shim_graph, tmp_path, monkeypatch):
    import os
    seed_cache(tmp_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    class GuardedTools(FakeTools):
        def index(self, paper_id, options):
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
            return super().index(paper_id, options)
    result = run_agent_research(QUESTION, paper_id=PAPER,
        options=ExecutionOptions(data_dir=tmp_path), tools=GuardedTools())
    assert result["status"] == "evidence_retrieved_no_answer"
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_preparation_stages_run_in_dependency_order_with_cached_local_pdf(shim_graph, tmp_path):
    paths = _paths(tmp_path, PAPER)
    paths["pdf"].parent.mkdir(parents=True, exist_ok=True)
    paths["pdf"].write_bytes(b"%PDF- synthetic placeholder for fake tools")
    tool = FakeTools()
    result = run_agent_research(QUESTION, paper_id=PAPER,
        options=ExecutionOptions(data_dir=tmp_path), tools=tool)
    assert result["status"] == "evidence_retrieved_no_answer"
    assert tool.calls[:4] == ["ingest", "chunk", "embed", "index"]
    assert result["tool_calls"] == 7
    assert result["plan"][-1]["id"] == "answer"
    assert result["plan"][-1]["execution"] == "not_executed"
    assert next(p["execution"] for p in result["plan"] if p["id"] == "prepare") == "completed"


def test_real_adapter_rejects_unverified_saved_embedding_bundle(tmp_path):
    seed_cache(tmp_path)
    with pytest.raises(ValueError):
        PaperTrailTools().index(PAPER, ExecutionOptions(data_dir=tmp_path))


def test_all_queries_without_results_are_reported_as_no_hits(shim_graph, tmp_path):
    seed_cache(tmp_path)
    class EmptyRetriever(FakeTools):
        def retrieve(self, paper_id, query, options):
            self._record("retrieve")
            return []
    result = run_agent_research(QUESTION, paper_id=PAPER,
        options=ExecutionOptions(data_dir=tmp_path), tools=EmptyRetriever())
    assert result["status"] == "no_retrieval_hits"
    assert result["retrieval_steps_completed"] == 3
    assert result["primary_evidence_hits"] == 0
    assert result["plan"][-2]["execution"] == "not_executed"
    assert result["plan"][-1]["execution"] == "not_executed"


def test_source_audit_rejects_chunks_not_equal_to_original_pdf_page(tmp_path):
    import pymupdf
    from papertrail.ingestion.pdf_extractor import extract_pdf
    from papertrail.chunking.chunker import chunk_document
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Testing exact page to chunk offsets with a long and informative research passage")
    blob = doc.tobytes()
    doc.close()
    paths = _paths(tmp_path, PAPER)
    for name in ("pdf", "extracted", "chunks"):
        paths[name].parent.mkdir(parents=True, exist_ok=True)
    pages = extract_pdf(blob, PAPER)
    paths["pdf"].write_bytes(blob)
    paths["extracted"].write_text(pages.model_dump_json(), encoding="utf-8")
    chunk_doc = chunk_document(pages, chunk_size=100, overlap=10)
    assert chunk_doc.chunks
    paths["chunks"].write_text(chunk_doc.model_dump_json(), encoding="utf-8")
    assert _source_audit(PAPER, tmp_path)["status"] == "pdf_extracted_hash_matched"
    tampered = chunk_doc.model_copy(deep=True)
    tampered.chunks[0].text = tampered.chunks[0].text.replace("Testing", "Hacking")
    paths["chunks"].write_text(tampered.model_dump_json(), encoding="utf-8")
    with pytest.raises(ExecutionError, match="offsets"):
        _source_audit(PAPER, tmp_path)


def test_adapter_arxiv_discovery_uses_bounded_safe_keyword_expression(monkeypatch):
    from types import SimpleNamespace
    from papertrail.discovery import arxiv_client
    class FakeArxiv:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] <= 5
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def search(self, expression, *, max_results):
            assert expression == "all:faithful AND all:retrieval AND all:augmented AND all:generation"
            assert max_results == 3
            return SimpleNamespace(papers=[SimpleNamespace(
                paper_id=PAPER, title="FAIR-RAG", url=f"https://arxiv.org/abs/{PAPER}")])
    monkeypatch.setattr(arxiv_client, "ArxivClient", FakeArxiv)
    out = PaperTrailTools().discover("What are recent methods for faithful retrieval augmented generation?",
        ExecutionOptions(allow_network=True, max_papers=3))
    assert out == [{"paper_id": PAPER, "title": "FAIR-RAG", "url": f"https://arxiv.org/abs/{PAPER}"}]


def test_focused_style_changes_only_retrieval_queries_in_native_equivalent_shim(shim_graph, tmp_path):
    seed_cache(tmp_path)
    base = FakeTools()
    focused = FakeTools()
    original = run_agent_research(QUESTION, paper_id=PAPER,
                                  options=ExecutionOptions(data_dir=tmp_path, max_refinement_rounds=0),
                                  tools=base, synthesize_mode=True)
    candidate = run_agent_research(QUESTION, paper_id=PAPER,
                                   options=ExecutionOptions(data_dir=tmp_path, query_style="focused",
                                                            max_refinement_rounds=0),
                                   tools=focused, synthesize_mode=True)
    assert original["query_style"] == "baseline"
    assert candidate["query_style"] == "focused"
    assert original["llm_calls"] == candidate["llm_calls"] == 0
    assert original["network_opt_in"] is candidate["network_opt_in"] is False
    assert [entry["query"] for entry in original["evidence"]] != [entry["query"] for entry in candidate["evidence"]]
    assert original["retrieval_steps_completed"] == candidate["retrieval_steps_completed"] == 3


def test_focused_strategy_cannot_invoke_live_synthesis_or_llm_planner(shim_graph, tmp_path):
    seed_cache(tmp_path)
    with pytest.raises(ExecutionError, match="diagnostic only"):
        run_agent_research(QUESTION, paper_id=PAPER,
                           options=ExecutionOptions(data_dir=tmp_path, query_style="focused"),
                           tools=FakeTools(), synthesize_mode=True,
                           synthesis_provider=lambda _: {"claims": []})
    with pytest.raises(ExecutionError, match="diagnostic only"):
        run_agent_research(QUESTION, paper_id=PAPER,
                           options=ExecutionOptions(data_dir=tmp_path, query_style="focused"),
                           tools=FakeTools(), planner_mode="llm")


def test_multi_query_shim_executes_two_validated_branches_each_with_bounded_evidence(shim_graph, tmp_path):
    seed_cache(tmp_path)

    class DualTools(FakeTools):
        def __init__(self):
            super().__init__()
            self.queries = []

        def retrieve(self, paper_id, query, options):
            self._record("retrieve")
            self.queries.append(query)
            step = (len(self.queries) - 1) // 2
            style = (len(self.queries) - 1) % 2
            names = ([f"common_{step}", f"base_{step}"] if not style
                     else [f"common_{step}", f"special_{step}"])
            return [{"chunk_id": name, "page_number": 10 + step,
                     "citation": f"https://arxiv.org/abs/{paper_id}#page={10+step}",
                     "rank": rank, "rrf_score": 0.025,
                     "text": f"Local fixture text {name}", "context": []}
                    for rank, name in enumerate(names, start=1)]

    adapter = DualTools()
    result = run_agent_research(
        QUESTION, paper_id=PAPER,
        options=ExecutionOptions(data_dir=tmp_path, query_style="multi", top_k=3,
                                 max_refinement_rounds=0),
        tools=adapter, synthesize_mode=True,
    )
    assert result["status"] == "ready_for_llm_synthesis"
    assert result["llm_calls"] == 0 and result["network_opt_in"] is False
    assert result["answer_generated"] is False and not result["errors"]
    assert result["tool_calls"] == 7  # 1 index + 2 x 3 retrievals
    assert adapter.calls == ["index"] + ["retrieve"] * 6
    assert len(result["evidence"]) == 3
    assert all(len(entry["query_traces"]) == 2 for entry in result["evidence"])
    assert all([branch["style"] for branch in entry["query_traces"]] == ["baseline", "focused"]
               for entry in result["evidence"])
    assert all(2 <= len(entry["hits"]) <= 3 for entry in result["evidence"])
    assert all(len({h["chunk_id"] for h in entry["hits"]}) == len(entry["hits"])
               for entry in result["evidence"])
    assert len(result["synthesis"]["source_register"]) == 9
    assert result["plan"][2]["search_query"] == result["evidence"][0]["query"]


def test_multi_query_shim_fails_closed_when_source_disagrees_and_when_budget_exhausted(shim_graph, tmp_path):
    seed_cache(tmp_path)

    class Contradiction(FakeTools):
        def __init__(self):
            super().__init__()
            self.queries = []

        def retrieve(self, paper_id, query, options):
            self.queries.append(query)
            return [{"chunk_id": "same_id", "page_number": 10,
                     "citation": f"https://arxiv.org/abs/{paper_id}#page=10",
                     "rank": 1, "rrf_score": 0.03,
                     "text": "conflicting duplicate" if len(self.queries) == 2 else "original",
                     "context": []}]

    tool = Contradiction()
    bad = run_agent_research(QUESTION, paper_id=PAPER,
                             options=ExecutionOptions(data_dir=tmp_path, query_style="multi",
                                                      max_refinement_rounds=0),
                             tools=tool, synthesize_mode=True)
    assert bad["status"] == "tool_failed"
    assert any(e["name"] == "multi_query_fusion" and e["status"] == "failed"
               for e in bad["tool_events"])
    assert bad["llm_calls"] == 0 and not bad["answer_generated"]
    budget = run_agent_research(QUESTION, paper_id=PAPER,
                                options=ExecutionOptions(data_dir=tmp_path, query_style="multi",
                                                         max_refinement_rounds=0, max_tool_calls=6),
                                tools=FakeTools(), synthesize_mode=True)
    assert budget["status"] == "tool_budget_exhausted"
    assert not budget["answer_generated"]
    assert budget["llm_calls"] == 0

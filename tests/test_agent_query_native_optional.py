"""Real StateGraph phase 4.2 integration on machines with the optional agent extra."""
from __future__ import annotations

import pytest
pytest.importorskip("langgraph", reason="Install [agent] to test real LangGraph focused query routing")

from papertrail.agent.evaluation import AgentEvaluationSuite
from papertrail.agent.query_eval import run_query_evaluation
from papertrail.agent.tool_execution import ExecutionOptions, _paths, run_agent_research

PAPER = "2510.22344"
QUESTION = "Compare FAIR-RAG with traditional RAG and discuss evaluation metrics and limitations."


class FixtureTools:
    def __init__(self):
        self.queries = []

    def discover(self, *_):
        raise AssertionError("Network not allowed")

    def ingest(self, *_):
        raise AssertionError("Cached artifacts required")

    def chunk(self, *_):
        raise AssertionError("Cached artifacts required")

    def embed(self, *_):
        raise AssertionError("Cached artifacts required")

    def index(self, *_):
        return {"cached": True, "vectors": 6}

    def retrieve(self, paper_id, query, options):
        self.queries.append(query)
        i = (len(self.queries) - 1) % 3 + 1
        return [{"chunk_id": f"synthetic_phase42_anchor_{i:02d}", "page_number": i + 1,
                 "citation": f"https://arxiv.org/abs/{PAPER}#page={i+1}",
                 "rank": 1, "rrf_score": 0.04,
                 "text": "Fixture only. Not scientific evidence.", "context": []},
                {"chunk_id": f"synthetic_phase42_other_{i:02d}", "page_number": i + 1,
                 "citation": f"https://arxiv.org/abs/{PAPER}#page={i+1}",
                 "rank": 2, "rrf_score": 0.03,
                 "text": "Fixture only. Not scientific evidence.", "context": []}]


def test_real_langgraph_paired_query_paths(tmp_path):
    for key in ("chunks", "vectors", "embedding_manifest"):
        path = _paths(tmp_path, PAPER)[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture-only")
    suite = AgentEvaluationSuite.model_validate({
        "suite_id": "native_query_smoke", "annotation_status": "provisional_log_derived_unverified",
        "provenance": "Synthetic integration references with no semantic relevance labels",
        "cases": [{"case_id": "native_case", "paper_id": PAPER, "question": QUESTION,
                   "facets": [{"step_id": f"retrieve_{i:02d}",
                               "anchor_chunk_ids": [f"synthetic_phase42_anchor_{i:02d}"],
                               "rationale": "Synthetic fixture reference, not human verified"}
                              for i in (1, 2, 3)]}],
    })
    observed = []

    def runner(question, paper_id, k, style):
        adapter = FixtureTools()
        result = run_agent_research(
            question, paper_id=paper_id,
            options=ExecutionOptions(data_dir=tmp_path, top_k=k, query_style=style,
                                     max_refinement_rounds=0),
            tools=adapter, synthesize_mode=True, synthesis_provider=None,
        )
        observed.append((style, list(adapter.queries)))
        return result

    report = run_query_evaluation(suite, data_dir=tmp_path, k_values=(3,), runner=runner)
    assert len(observed) == 2
    assert observed[0][0] == "baseline"
    assert observed[1][0] == "focused"
    assert observed[0][1] != observed[1][1]
    assert report["llm_calls"] == 0
    assert report["comparisons"][0]["delta_observed_anchor_fraction"] == 0.0
    assert all(run["anchors_retrieved"] == 3 for run in report["per_run"])

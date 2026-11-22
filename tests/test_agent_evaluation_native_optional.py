"""Native LangGraph Phase 4.1 smoke test; runs on Windows with [agent] installed."""
import pytest

pytest.importorskip("langgraph", reason="Install [agent] to run native Phase 4.1 evaluation integration")

from papertrail.agent.evaluation import AgentEvaluationSuite, run_agent_evaluation
from papertrail.agent.tool_execution import ExecutionOptions, _paths, run_agent_research

PAPER = "2510.22344"
Q = "Compare FAIR-RAG with traditional RAG and discuss evaluation metrics and limitations."


class FakeTools:
    def __init__(self):
        self.n = 0

    def discover(self, *_):
        raise AssertionError("No network")

    def ingest(self, *_):
        raise AssertionError("Cache expected")

    def chunk(self, *_):
        raise AssertionError("Cache expected")

    def embed(self, *_):
        raise AssertionError("Cache expected")

    def index(self, *_):
        return {"cached": True, "vectors": 6}

    def retrieve(self, paper_id, query, options):
        self.n += 1
        name = (self.n - 1) % 3 + 1
        # Deterministic matched chunk ID for each evaluation facet.
        return [{"chunk_id": chunk_id, "page_number": name + 2,
                 "citation": f"https://arxiv.org/abs/{PAPER}#page={name + 2}",
                 "rank": rank, "rrf_score": 0.04, "text": "This synthetic passage is for offline native integration testing only.",
                 "context": []}
                for rank, chunk_id in enumerate(
                    (f"native_anchor_facet_{name:02d}", f"native_secondary_facet_{name:02d}"), 1)]


def test_real_langgraph_preview_pipeline_evaluation(tmp_path):
    for key in ("chunks", "vectors", "embedding_manifest"):
        path = _paths(tmp_path, PAPER)[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-only fixture")
    suite = AgentEvaluationSuite.model_validate({
        "suite_id": "native_regression", "annotation_status": "provisional_log_derived_unverified",
        "provenance": "Synthetic native pipeline contract test, not factual gold annotations",
        "cases": [{"case_id": "native_case", "paper_id": PAPER, "question": Q,
                   "facets": [{"step_id": f"retrieve_{n:02d}",
                               "anchor_chunk_ids": [f"native_anchor_facet_{n:02d}"],
                               "rationale": "Synthetic fixture target for native integration test"}
                              for n in (1, 2, 3)]}],
    })

    def runner(question, paper_id, k):
        return run_agent_research(question, paper_id=paper_id,
                                  options=ExecutionOptions(data_dir=tmp_path, top_k=k, max_refinement_rounds=0),
                                  tools=FakeTools(), synthesize_mode=True, synthesis_provider=None)
    result = run_agent_evaluation(suite, data_dir=tmp_path, k_values=(3, 5), runner=runner)
    assert result["llm_calls"] == 0
    assert all(row["anchors_retrieved"] == 3 for row in result["per_run"])
    assert all(row["anchors_model_visible"] == 3 for row in result["per_run"])

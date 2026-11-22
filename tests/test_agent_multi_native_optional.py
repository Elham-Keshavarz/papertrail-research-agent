"""Native LangGraph smoke for Phase 4.2.1; runs on user machines with [agent]."""
from __future__ import annotations

import pytest
pytest.importorskip("langgraph", reason="Install [agent] to run native multi-query StateGraph integration")

from papertrail.agent.evaluation import AgentEvaluationSuite
from papertrail.agent.multi_eval import run_multi_query_evaluation
from papertrail.agent.tool_execution import ExecutionOptions, _paths, run_agent_research

PAPER = "2510.22344"
QUESTION = "Compare FAIR-RAG with traditional RAG and discuss evaluation metrics and limitations."


class FixtureTools:
    def __init__(self):
        self.queries = []

    def discover(self, *_):
        raise AssertionError("Network request is prohibited")

    def ingest(self, *_):
        raise AssertionError("Fixture cache missing")

    def chunk(self, *_):
        raise AssertionError("Fixture cache missing")

    def embed(self, *_):
        raise AssertionError("Fixture cache missing")

    def index(self, *_):
        return {"cached": True, "vectors": 6}

    def retrieve(self, paper_id, query, options):
        self.queries.append(query)
        index = (len(self.queries)-1) // (2 if options.query_style in {"multi", "budgeted"} else 1)
        names = [f"fixture_{index}_anchor", f"fixture_{index}_other"]
        return [{"chunk_id": chunk, "page_number": index+2,
                 "citation": f"https://arxiv.org/abs/{paper_id}#page={index+2}",
                 "rank": rank, "rrf_score": 0.04, "text": "Synthetic text only.", "context": []}
                for rank, chunk in enumerate(names, start=1)]


def test_native_langgraph_three_way_multi_query_benchmark(tmp_path):
    for item in ("chunks", "vectors", "embedding_manifest"):
        path = _paths(tmp_path, PAPER)[item]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture-only")
    suite = AgentEvaluationSuite.model_validate({
        "suite_id": "native_multi_smoke", "annotation_status": "provisional_log_derived_unverified",
        "provenance": "Synthetic fixed integration references, not an external truth benchmark",
        "cases": [{"case_id": "native_case", "paper_id": PAPER, "question": QUESTION,
                   "facets": [{"step_id": f"retrieve_{i:02d}",
                               "anchor_chunk_ids": [f"fixture_{i-1}_anchor"],
                               "rationale": "Synthetic and not independently verified"}
                              for i in (1, 2, 3)]}],
    })
    queries = []

    def runner(question, paper_id, k, style):
        tool = FixtureTools()
        result = run_agent_research(
            question, paper_id=paper_id,
            options=ExecutionOptions(data_dir=tmp_path, top_k=k, query_style=style,
                                     max_refinement_rounds=0),
            tools=tool, synthesize_mode=True, synthesis_provider=None,
        )
        queries.append((style, list(tool.queries)))
        return result

    report = run_multi_query_evaluation(suite, data_dir=tmp_path, k_values=(3,), runner=runner)
    assert report["run_count"] == 3
    assert all(item["preview_status"] == "ready_for_llm_synthesis" for item in report["per_run"])
    assert report["comparisons"][0]["multi_anchors_post_budget"] == 3
    assert report["comparisons"][0]["multi_anchors_model_visible"] == 3
    assert [len(qs) for _, qs in queries] == [3, 3, 6]
    assert all(item["llm_calls"] == 0 for item in report["per_run"])


def test_native_langgraph_budget_aware_ablation(tmp_path):
    """Run all four styles through the real graph with synthetic offline tools."""
    from papertrail.agent.budget_eval import run_budget_evaluation
    for item in ("chunks", "vectors", "embedding_manifest"):
        path = _paths(tmp_path, PAPER)[item]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture-only")
    suite = AgentEvaluationSuite.model_validate({
        "suite_id": "native_budget_smoke", "annotation_status": "provisional_log_derived_unverified",
        "provenance": "Synthetic integration smoke, not relevance labels",
        "cases": [{"case_id": "native_case", "paper_id": PAPER, "question": QUESTION,
                   "facets": [{"step_id": f"retrieve_{i:02d}",
                               "anchor_chunk_ids": [f"fixture_{i-1}_anchor"],
                               "rationale": "Synthetic and not independently verified"}
                              for i in (1, 2, 3)]}],
    })
    counts = []
    def runner(question, paper_id, k, style):
        tools = FixtureTools()
        result = run_agent_research(question, paper_id=paper_id,
                                    options=ExecutionOptions(data_dir=tmp_path, top_k=k, query_style=style,
                                                             max_refinement_rounds=0),
                                    tools=tools, synthesize_mode=True, synthesis_provider=None)
        counts.append(len(tools.queries))
        return result
    report = run_budget_evaluation(suite, data_dir=tmp_path, k_values=(3,), runner=runner)
    assert report["run_count"] == 4
    assert counts == [3, 3, 6, 6]
    assert report["llm_calls"] == 0
    assert report["comparisons"][0]["retrieval_query_counts"]["budgeted"] == 6
    assert all(row["preview_status"] == "ready_for_llm_synthesis" for row in report["per_run"])

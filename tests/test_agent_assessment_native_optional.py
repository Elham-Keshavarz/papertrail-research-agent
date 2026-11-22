"""Native LangGraph Phase 3.4 rerouting regression (requires [agent] extra)."""
import pytest

pytest.importorskip("langgraph", reason="Install [agent] to run native evidence-assessment workflow tests")

from papertrail.agent.tool_execution import ExecutionOptions, _paths, run_agent_research

PAPER = "2510.22344"
QUESTION = "Compare FAIR-RAG with traditional RAG and discuss evaluation metrics."


class Adapter:
    def __init__(self):
        self.calls, self.queries = [], []

    def index(self, *_):
        self.calls.append("index")
        return {"cached": True, "vectors": 4}

    def retrieve(self, _paper, query, _options):
        self.calls.append("retrieve")
        self.queries.append(query)
        label = "retry" if query == QUESTION else "initial"
        return [{"chunk_id": label, "rank": 1, "rrf_score": 0.03,
                 "page_number": 4, "text": "Synthetic evidence only",
                 "context": [], "citation": f"https://arxiv.org/abs/{PAPER}#page=4"}]

    def discover(self, *_):
        raise AssertionError("Unexpected network access")

    def ingest(self, *_):
        raise AssertionError("Unexpected PDF re-ingest")

    def chunk(self, *_):
        raise AssertionError("Unexpected chunking")

    def embed(self, *_):
        raise AssertionError("Unexpected embedding")


def test_native_assessment_routes_through_retries(tmp_path):
    paths = _paths(tmp_path, PAPER)
    for name in ("chunks", "vectors", "embedding_manifest"):
        paths[name].parent.mkdir(exist_ok=True, parents=True)
        paths[name].write_bytes(b"fake data")
    fake = Adapter()
    result = run_agent_research(QUESTION, paper_id=PAPER,
        options=ExecutionOptions(data_dir=tmp_path), tools=fake, assess_mode=True)
    assert result["status"] == "evidence_assessed_unverified"
    assert result["assessment"]["subquestions_meeting_threshold"] == result["assessment"]["planned_subquestions"]
    assert result["tool_calls"] == 1 + len(fake.queries)
    assert not result["answer_generated"]
    assert result["plan"][-2]["execution"] == "completed_structural_unverified"
    assert result["plan"][-1]["execution"] == "not_executed"

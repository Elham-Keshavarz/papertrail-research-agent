"""Native LangGraph conditional routes: executed when [agent] extra is installed."""
import pytest

pytest.importorskip("langgraph", reason="Install [agent] to run native tool-execution tests")

from papertrail.agent.tool_execution import ExecutionOptions, run_agent_research

PAPER = "2510.22344"
QUESTION = "Compare FAIR-RAG and traditional RAG on evaluation metrics and limitations."


class NativeFakeTools:
    def __init__(self):
        self.calls = []

    def discover(self, question, options):
        self.calls.append("discover")
        return [{"paper_id": PAPER, "title": "FAIR-RAG"}]

    def ingest(self, paper_id, options):
        self.calls.append("ingest")
        return {"cached_pdf": True}

    def chunk(self, paper_id, options):
        self.calls.append("chunk")
        return {"chunks": 1}

    def embed(self, paper_id, options):
        self.calls.append("embed")
        return {"vectors": 1}

    def index(self, paper_id, options):
        self.calls.append("index")
        return {"cached": True, "vectors": 1}

    def retrieve(self, paper_id, query, options):
        self.calls.append("retrieve")
        return [{"chunk_id": "native-a", "page_number": 2,
                 "citation": "https://arxiv.org/abs/2510.22344#page=2", "rank": 1,
                 "rrf_score": 0.031, "text": "Synthetic fixture (not a real claim)", "context": []}]


def _seed(tmp_path):
    from papertrail.agent.tool_execution import _paths
    paths = _paths(tmp_path, PAPER)
    for name in ("chunks", "vectors", "embedding_manifest"):
        paths[name].parent.mkdir(exist_ok=True, parents=True)
        paths[name].write_bytes(b"fixture")


def test_native_langgraph_executes_per_subquestion_and_finalizes(tmp_path):
    _seed(tmp_path)
    fake = NativeFakeTools()
    result = run_agent_research(QUESTION, paper_id=PAPER,
        options=ExecutionOptions(data_dir=tmp_path), tools=fake)
    assert result["status"] == "evidence_retrieved_no_answer"
    assert result["tool_calls"] == 4
    assert fake.calls == ["index", "retrieve", "retrieve", "retrieve"]
    assert result["trace"][-1]["node"] == "finalize"
    assert not result["answer_generated"]


def test_native_langgraph_offline_missing_paper_selection_no_network(tmp_path):
    fake = NativeFakeTools()
    result = run_agent_research(QUESTION, options=ExecutionOptions(data_dir=tmp_path), tools=fake)
    assert result["status"] == "needs_paper_selection"
    assert fake.calls == []


def test_native_langgraph_bounded_budget_no_extra_calls(tmp_path):
    _seed(tmp_path)
    fake = NativeFakeTools()
    result = run_agent_research(QUESTION, paper_id=PAPER,
        options=ExecutionOptions(data_dir=tmp_path, max_tool_calls=2), tools=fake)
    assert result["status"] == "tool_budget_exhausted"
    assert fake.calls == ["index", "retrieve"]
    assert result["tool_calls"] == 2


def test_native_streamed_progress_preserves_research_and_tool_budget(tmp_path):
    """Browser progress must not duplicate retrieval or change the original result."""
    _seed(tmp_path)
    no_progress_tools = NativeFakeTools()
    options = ExecutionOptions(data_dir=tmp_path)
    expected = run_agent_research(QUESTION, paper_id=PAPER,
                                  options=options, tools=no_progress_tools)

    streamed_tools = NativeFakeTools()
    checkpoints = []
    actual = run_agent_research(QUESTION, paper_id=PAPER,
                                options=options, tools=streamed_tools,
                                on_progress=lambda percentage, label: checkpoints.append((percentage, label)))
    assert actual['status'] == expected['status']
    assert actual['evidence'] == expected['evidence']
    assert actual['tool_calls'] == expected['tool_calls'] == 4
    assert streamed_tools.calls == no_progress_tools.calls
    assert checkpoints[0][0] == 0
    assert checkpoints[-1][0] == 100
    assert any('Retrieved evidence for 1/' in label for _, label in checkpoints)
    assert [percent for percent, _ in checkpoints] == sorted(percent for percent, _ in checkpoints)

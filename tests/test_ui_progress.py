"""Progress follows completed workflow nodes, never fake elapsed time."""
import pytest

from papertrail.agent.progress import research_checkpoint
from papertrail.agent import tool_execution as te


def state(node, **kwargs):
    return {'trace': [{'node': node, 'event': 'completed'}], **kwargs}


def test_real_node_checkpoint_percentages_and_messages():
    assert research_checkpoint({'trace': []}) is None
    assert research_checkpoint(state('plan_research')) == (15, 'Research question broken into search tasks')
    assert research_checkpoint(state('prepare')) == (35, 'Local source and index checked')
    result = research_checkpoint(state('retrieve', plan=[
        {'proposed_tool': 'hybrid_retrieval'} for _ in range(3)
    ], evidence=[{'hits': []}, {'hits': []}]))
    assert result == (61, 'Retrieved evidence for 2/3 research tasks')
    assert research_checkpoint(state('synthesize_answer'))[0] == 95
    assert research_checkpoint(state('finalize'))[0] == 100
    assert research_checkpoint(state('made_up_node')) is None


def test_stream_callbacks_follow_nodes_and_invoke_default_unchanged(monkeypatch, tmp_path):
    output = {
        'status': 'ready_for_llm_synthesis', 'paper_id': '2510.22344',
        'question_analysis': {'intents': []}, 'plan': [
            {'proposed_tool': 'hybrid_retrieval', 'id': 'retrieve_01', 'execution': 'pending'},
            {'proposed_tool': 'hybrid_retrieval', 'id': 'retrieve_02', 'execution': 'pending'},
        ],
        'evidence': [{'step_id': 'retrieve_01', 'hits': []},
                     {'step_id': 'retrieve_02', 'hits': []}],
        'tool_events': [], 'tool_calls': 2, 'synthesis': {'answer': None},
    }
    class FakeGraph:
        def __init__(self):
            self.called_invoke = False
            self.called_stream = False
        def invoke(self, initial, **kwargs):
            self.called_invoke = True
            return {**output, 'trace': [{'node': 'finalize'}]}
        def stream(self, initial, **kwargs):
            self.called_stream = True
            assert kwargs['stream_mode'] == 'values'
            yield {**initial}
            yield {**output, 'trace': [{'node': 'prepare'}], 'evidence': []}
            yield {**output, 'trace': [{'node': 'retrieve'}], 'evidence': output['evidence'][:1]}
            yield {**output, 'trace': [{'node': 'retrieve'}]}
            yield {**output, 'trace': [{'node': 'synthesize_answer'}]}
            yield {**output, 'trace': [{'node': 'finalize'}]}
    graph = FakeGraph()
    monkeypatch.setattr(te, 'build_tool_graph', lambda *args, **kwargs: graph)
    options = te.ExecutionOptions(data_dir=tmp_path)
    events = []
    live = te.run_agent_research('What does the paper describe?', paper_id='2510.22344',
                                 options=options, synthesize_mode=True,
                                 on_progress=lambda p, label: events.append((p, label)))
    assert not graph.called_invoke and graph.called_stream
    assert live['status'] == 'ready_for_llm_synthesis'
    assert [p for p, _ in events] == [0, 35, 55, 75, 95, 100]
    graph2 = FakeGraph()
    monkeypatch.setattr(te, 'build_tool_graph', lambda *args, **kwargs: graph2)
    te.run_agent_research('What does the paper describe?', paper_id='2510.22344', options=options)
    assert graph2.called_invoke and not graph2.called_stream


def test_stream_progress_does_not_complete_when_execution_raises(monkeypatch, tmp_path):
    class BrokenGraph:
        def stream(self, initial, **kwargs):
            yield {**initial, 'trace': [{'node': 'prepare'}]}
            raise RuntimeError('simulated failure')
    monkeypatch.setattr(te, 'build_tool_graph', lambda *args, **kwargs: BrokenGraph())
    events = []
    with pytest.raises(RuntimeError):
        te.run_agent_research('What does the paper describe?', paper_id='2510.22344',
            options=te.ExecutionOptions(data_dir=tmp_path),
            on_progress=lambda p, label: events.append(p))
    assert events == [0, 35]

"""Native LangGraph integration; runs on Windows when `[agent]` is installed."""
import pytest

pytest.importorskip('langgraph', reason='Install [agent] to test native cited-answer routing')

from papertrail.agent.tool_execution import ExecutionOptions, _paths, run_agent_research

PAPER = '2510.22344'
Q = 'Compare FAIR-RAG with ordinary RAG on evaluation metrics and limitations.'
PHRASE = 'This is a synthetic source passage for native graph smoke testing only.'


class FakeTools:
    def discover(self, *_):
        raise AssertionError('No remote calls')
    def ingest(self, *_):
        raise AssertionError('Should use cache')
    def chunk(self, *_):
        raise AssertionError('Should use cache')
    def embed(self, *_):
        raise AssertionError('Should use cache')
    def index(self, *_):
        return {'cached': True, 'vectors': 2}
    def retrieve(self, paper_id, query, options):
        return [
            {'chunk_id': f'{query[:8]}-{n}', 'page_number': 3 + n,
             'citation': f'https://arxiv.org/abs/{PAPER}#page={3+n}',
             'rank': n, 'rrf_score': 0.04, 'text': PHRASE + f' Evidence {n}.', 'context': []}
            for n in (1, 2)
        ]


def _seed(tmp_path):
    paths = _paths(tmp_path, PAPER)
    for k in ('chunks', 'vectors', 'embedding_manifest'):
        paths[k].parent.mkdir(parents=True, exist_ok=True)
        paths[k].write_bytes(b'synthetic fixture')


def test_native_graph_quote_validated_claims(tmp_path):
    _seed(tmp_path)
    requests = []
    def model(payload):
        requests.append(payload)
        return {'claims': [
            {'step_id': facet['step_id'],
             'claim': f'This synthetic claim concerns facet {facet["step_id"]} only.',
             'citations': [{'evidence_id': next(e for e in payload['evidence'] if e['step_id'] == facet['step_id'])['evidence_id'],
                            'quote': PHRASE}]}
            for facet in payload['facets']
        ]}
    result = run_agent_research(Q, paper_id=PAPER,
        options=ExecutionOptions(data_dir=tmp_path, max_refinement_rounds=0),
        tools=FakeTools(), synthesize_mode=True, synthesis_provider=model)
    assert result['status'] == 'cited_answer_unverified'
    assert result['llm_calls'] == 1 and result['answer_generated']
    assert len(requests) == 1
    assert result['trace'][-2]['node'] == 'synthesize_answer'
    assert result['trace'][-1]['node'] == 'finalize'


def test_native_preview_never_invokes_external_model(tmp_path):
    _seed(tmp_path)
    result = run_agent_research(Q, paper_id=PAPER,
        options=ExecutionOptions(data_dir=tmp_path, max_refinement_rounds=0),
        tools=FakeTools(), synthesize_mode=True)
    assert result['status'] == 'ready_for_llm_synthesis'
    assert result['llm_calls'] == 0 and result['answer'] is None

"""Phase 3.5: validation of untrusted model outputs and real graph routing (offline)."""
import json
import sys
import types

import pytest

from papertrail.agent.synthesis import (
    SynthesisError, build_evidence_register, make_synthesis_payload,
    validate_and_render_draft, validate_with_partial_recovery, OpenAICompatibleAnswerGenerator,
)
from papertrail.agent.tool_execution import ExecutionOptions, _paths, run_agent_research

PAPER = '2510.22344'
Q = 'Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations.'
PHRASE = 'The evaluation covered multiple question answering benchmarks with clear measurements.'


def _plan():
    return [{'id': f'retrieve_0{i}', 'proposed_tool': 'hybrid_retrieval',
             'search_query': f'subquestion {i}', 'execution': 'not_executed'} for i in range(1, 4)]


def _hit(sid, n, page=None, phrase=PHRASE):
    page = page or 10 + n
    return {'chunk_id': f'{sid}-hit-{n}', 'page_number': page, 'rank': n, 'rrf_score': 0.03,
            'citation': f'https://arxiv.org/abs/{PAPER}#page={page}',
            'text': phrase + f' Evidence for {sid} number {n}.', 'context': []}


def _data():
    plan = _plan()
    evidence = [{'step_id': s['id'], 'hits': [_hit(s['id'], 1), _hit(s['id'], 2)]} for s in plan]
    register = build_evidence_register(plan, evidence, PAPER)
    return plan, evidence, register


def _claims(register, omit=None):
    taken = {}
    for r in register:
        taken.setdefault(r['step_id'], r)
    return {'claims': [
        {'step_id': sid, 'claim': 'The evaluation covered multiple question answering benchmarks.',
         'citations': [{'evidence_id': src['evidence_id'], 'quote': PHRASE}]}
        for sid, src in taken.items() if sid != omit
    ]}


def test_register_uses_known_ids_paper_page_and_bounds():
    plan, evidence, r = _data()
    assert [x['evidence_id'] for x in r] == [f'E{i:02d}' for i in range(1, 7)]
    assert r[0]['page_number'] == 11
    assert r[0]['citation'] == f'https://arxiv.org/abs/{PAPER}#page=11'
    assert len(make_synthesis_payload(Q, plan, r)['evidence']) == 6


def test_valid_quote_anchored_claims_assembled_from_source_ids():
    plan, _, r = _data()
    # Distinct claims across facets (the model needs to actually differentiate).
    d = _claims(r)
    for i, c in enumerate(d['claims']):
        c['claim'] += f' Research aspect {chr(ord("A") + i)}.'
    v = validate_and_render_draft(d, plan, r)
    assert v['status'] == 'cited_answer_unverified'
    assert v['claim_count'] == 3
    assert '[E01](https://arxiv.org/abs/2510.22344#page=11)' in v['answer']
    assert not v['semantic_entailment_verified'] and v['human_review_required']


@pytest.mark.parametrize('tamper', ['unknown_id', 'bad_page', 'fake_quote', 'cross_facet',
                                  'numeric_hallucination', 'markup', 'unexpected_field', 'duplicate_reference'])
def test_model_forgery_or_unanchored_data_rejected(tamper):
    plan, _, r = _data()
    d = _claims(r, omit='retrieve_02')
    d['claims'][1]['claim'] += ' A second research aspect.'
    c = d['claims'][0]
    if tamper == 'unknown_id':
        c['citations'][0]['evidence_id'] = 'E999'
    elif tamper == 'bad_page':
        c['citations'][0]['evidence_id'] = 'https://attacker.invalid/#page=5'
    elif tamper == 'fake_quote':
        c['citations'][0]['quote'] = 'Completely fabricated evidence that the source never stated.'
    elif tamper == 'cross_facet':
        c['citations'][0]['evidence_id'] = 'E03'
    elif tamper == 'numeric_hallucination':
        c['claim'] += ' The model reported 92.6% accuracy.'
    elif tamper == 'markup':
        c['claim'] += ' [malicious](http://elsewhere.invalid)'
    elif tamper == 'unexpected_field':
        c['secret'] = 'unexpected'
    else:
        c['citations'].append(dict(c['citations'][0]))
    with pytest.raises(SynthesisError):
        validate_and_render_draft(d, plan, r)


def test_model_empty_claims_explicit_abstention():
    plan, _, r = _data()
    value = validate_and_render_draft({'claims': []}, plan, r)
    assert value['status'] == 'insufficient_evidence'
    assert value['answer'] is None
    assert len(value['missing_step_ids']) == 3


def test_model_partial_claims_not_mislabeled_complete():
    plan, _, r = _data()
    d = _claims(r, omit='retrieve_02')
    d['claims'][1]['claim'] += ' A second research aspect.'
    value = validate_and_render_draft(d, plan, r)
    assert value['status'] == 'partial_answer_unverified'
    assert value['missing_step_ids'] == ['retrieve_02']


def test_source_forgery_fails_before_model_call():
    p, evidence, _ = _data()
    evidence[0]['hits'][0]['citation'] = 'https://fake.invalid'
    with pytest.raises(SynthesisError):
        build_evidence_register(p, evidence, PAPER)


def test_large_input_fails_closed():
    p, _, r = _data()
    with pytest.raises(SynthesisError):
        make_synthesis_payload(Q * 600, p, r * 200)


def test_source_context_is_not_accepted_across_pages():
    p, evidence, _ = _data()
    evidence[0]['hits'][0]['context'] = [{
        'citation': f'https://arxiv.org/abs/{PAPER}#page=500', 'page_number': 500,
        'text': PHRASE,
    }]
    with pytest.raises(SynthesisError):
        build_evidence_register(p, evidence, PAPER)


@pytest.mark.parametrize('url', ['http://example.org/v1', 'https://someone:pw@example.org/v1',
                                  'https://169.254.1.1/v1', 'https://example.org/v1?key=hidden',
                                  'file:///tmp/my-endpoint'])
def test_unsafe_model_endpoints_rejected(url):
    with pytest.raises(SynthesisError):
        OpenAICompatibleAnswerGenerator(api_key='mock', model='fake-model', base_url=url)


def test_live_model_client_uses_single_bounded_request_and_never_logs_key(monkeypatch):
    calls = []
    class FakeResponse:
        def raise_for_status(self):
            return None
        def json(self):
            return {'choices': [{'message': {'content': json.dumps({'claims': []})}}]}
    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs['follow_redirects'] is False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def post(self, url, headers, json):
            calls.append((url, headers, json))
            return FakeResponse()
    monkeypatch.setattr('papertrail.agent.synthesis.httpx.Client', FakeClient)
    model = OpenAICompatibleAnswerGenerator(api_key='test-only-secret', model='test-model',
                                            base_url='https://api.example.com/v1')
    result = model({'question': Q, 'evidence': [], 'facets': []})
    assert result == {'claims': []} and len(calls) == 1
    assert calls[0][0] == 'https://api.example.com/v1/chat/completions'
    assert calls[0][2]['temperature'] == 0
    assert 'test-only-secret' not in json.dumps(calls[0][2])


def test_provider_error_does_not_leak_secret_body(monkeypatch):
    class ErrorClient:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, **kwargs):
            raise __import__('httpx').ConnectError('secret-in-provider-exception')
    monkeypatch.setattr('papertrail.agent.synthesis.httpx.Client', ErrorClient)
    model = OpenAICompatibleAnswerGenerator(api_key='private-credential', model='mock',
                                            base_url='https://api.example.com/v1')
    with pytest.raises(SynthesisError) as exc:
        model({'question': Q})
    assert 'private-credential' not in str(exc.value)
    assert 'secret-in-provider-exception' not in str(exc.value)


class MiniGraph:
    def __init__(self, _state):
        self.nodes, self.edges, self.routes = {}, {}, {}
    def add_node(self, name, fn):
        self.nodes[name] = fn
    def add_edge(self, src, dst):
        self.edges[src] = dst
    def add_conditional_edges(self, src, fn):
        self.routes[src] = fn
    def compile(self):
        return self
    def invoke(self, state, config=None):
        state = dict(state)
        node = self.edges['__start__']
        budget = 0
        while node != '__end__':
            budget += 1
            assert budget < 40
            updates = self.nodes[node](state)
            for key, value in updates.items():
                state[key] = state.get(key, []) + value if key == 'trace' else value
            node = self.routes[node](state) if node in self.routes else self.edges[node]
        return state


@pytest.fixture
def graph_shim(monkeypatch):
    stub = types.ModuleType('langgraph.graph')
    stub.StateGraph, stub.START, stub.END = MiniGraph, '__start__', '__end__'
    monkeypatch.setitem(sys.modules, 'langgraph', types.ModuleType('langgraph'))
    monkeypatch.setitem(sys.modules, 'langgraph.graph', stub)


class FakeTools:
    def __init__(self, hits=2):
        self.calls = []
        self.hits = hits
    def discover(self, *_):
        raise AssertionError('No discovery needed')
    def ingest(self, *_):
        raise AssertionError('Cached source')
    def chunk(self, *_):
        raise AssertionError('Cached chunks')
    def embed(self, *_):
        raise AssertionError('Cached vectors')
    def index(self, *_):
        self.calls.append('index')
        return {'cached': True, 'vectors': 421}
    def retrieve(self, paper_id, query, options):
        self.calls.append('retrieve')
        step = f'retrieve_0{self.calls.count("retrieve")}'
        return [_hit(step, i) for i in range(1, self.hits + 1)]


def seed(tmp_path):
    paths = _paths(tmp_path, PAPER)
    for name in ('chunks', 'vectors', 'embedding_manifest'):
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_bytes(b'fixture')


def execute(tmp_path, tools, provider, *, retry=0, **kwargs):
    seed(tmp_path)
    return run_agent_research(Q, paper_id=PAPER,
        options=ExecutionOptions(data_dir=tmp_path, max_refinement_rounds=retry, **kwargs),
        tools=tools, synthesize_mode=True, synthesis_provider=provider)


def test_graph_preview_is_zero_llm_and_no_answer(graph_shim, tmp_path):
    result = execute(tmp_path, FakeTools(), None)
    assert result['status'] == 'ready_for_llm_synthesis'
    assert result['llm_calls'] == 0
    assert result['answer'] is None and not result['answer_generated']
    assert result['plan'][-1]['execution'] == 'preview_not_executed'
    assert len(result['synthesis']['source_register']) == 6


def test_graph_live_accepts_cited_claims_from_fake_llm(graph_shim, tmp_path):
    requested = []
    def fake_llm(payload):
        requested.append(payload)
        claims = []
        for step in payload['facets']:
            found = next(e for e in payload['evidence'] if e['step_id'] == step['step_id'])
            claims.append({'step_id': step['step_id'],
                           'claim': f'Retrieved context for research facet {step["step_id"]} supports the source excerpt.',
                           'citations': [{'evidence_id': found['evidence_id'], 'quote': PHRASE}]})
        return {'claims': claims}
    result = execute(tmp_path, FakeTools(), fake_llm)
    assert result['status'] == 'cited_answer_unverified'
    assert result['answer_generated'] and len(result['synthesis']['claims']) == 3
    assert result['llm_calls'] == 1 and result['tool_calls'] == 4
    assert result['plan'][-1]['execution'] == 'completed_cited_unverified'
    assert len(requested) == 1
    assert result['human_review_required'] is True
    assert result['synthesis']['faithfulness_audit']['accepted_claim_count'] == 3
    assert result['synthesis']['faithfulness_audit']['quote_anchored_nonverbatim_count'] == 3
    assert result['synthesis']['faithfulness_audit']['semantic_entailment_verified'] is False


def test_graph_missing_facet_abstains_without_model_call(graph_shim, tmp_path):
    requested = []
    result = execute(tmp_path, FakeTools(hits=0), lambda payload: requested.append(payload), retry=0)
    assert result['status'] == 'insufficient_evidence'
    assert result['llm_calls'] == 0 and requested == []
    assert result['answer'] is None


def test_graph_model_failure_counts_one_call_redacts_error(graph_shim, tmp_path):
    def fail(payload):
        raise RuntimeError('secret-should-not-leak')
    result = execute(tmp_path, FakeTools(), fail)
    assert result['status'] == 'llm_unavailable'
    assert result['llm_calls'] == 1
    assert 'secret-should-not-leak' not in json.dumps(result)


def test_graph_rejects_hallucinated_claim_citation(graph_shim, tmp_path):
    def fake(payload):
        e = payload['evidence'][0]
        return {'claims': [{'step_id': e['step_id'], 'claim': 'The method achieved 99.9% verified accuracy.',
                            'citations': [{'evidence_id': e['evidence_id'], 'quote': PHRASE}]}]}
    result = execute(tmp_path, FakeTools(), fake)
    assert result['status'] == 'citation_validation_failed'
    assert result['llm_calls'] == 1
    assert result['answer'] is None and not result['answer_generated']


def test_tool_budget_exhaustion_prevents_model_call(graph_shim, tmp_path):
    calls = []
    result = execute(tmp_path, FakeTools(), lambda x: calls.append(x), max_tool_calls=2)
    assert result['status'] == 'tool_budget_exhausted'
    assert not calls and result['llm_calls'] == 0


def test_source_register_fairly_covers_five_facets():
    plan = [{'id': f'retrieve_0{i}', 'proposed_tool': 'hybrid_retrieval', 'search_query': f'facet {i}'} for i in range(1, 6)]
    evidence = [{'step_id': step['id'], 'hits': [_hit(step['id'], n) for n in range(1, 4)]} for step in plan]
    register = build_evidence_register(plan, evidence, PAPER)
    assert len(register) == 10
    assert {r['step_id'] for r in register} == {s['id'] for s in plan}


# Phase 3.5.1: no extra model requests, safe reason codes, partial recovery.
def test_diagnostic_wrong_quote_is_safe_and_indexed():
    plan, _, sources = _data()
    data = _claims(sources)
    for i, item in enumerate(data['claims']):
        item['claim'] += f' Aspect {chr(65 + i)}.'
    data['claims'][0]['citations'][0]['quote'] = 'This is not present in any real retrieved source passage.'
    checked = validate_with_partial_recovery(data, plan, sources)
    assert checked['reviewed']['status'] == 'partial_answer_unverified'
    assert checked['reviewed']['rejected_claims_count'] == 1
    assert checked['rejections'] == [{'claim_index': 1, 'reason_code': 'quote_not_exact_in_source'}]
    assert 'This is not present' not in json.dumps(checked)


def test_all_forged_claims_rejected_with_no_answer():
    plan, _, sources = _data()
    data = {'claims': [{'step_id': 'retrieve_01', 'claim': 'Method supposedly achieved 99.9 percent.',
                        'citations': [{'evidence_id': 'E999', 'quote': PHRASE}]}]}
    checked = validate_with_partial_recovery(data, plan, sources)
    assert checked['reviewed'] is None
    assert checked['rejections'] == [{'claim_index': 1, 'reason_code': 'citation_id_or_facet_invalid'}]
    assert 'E999' not in json.dumps(checked)


def test_oversized_draft_cannot_evade_limit_via_salvage():
    plan, _, sources = _data()
    data = {'claims': [_claims(sources)['claims'][0]] * 40}
    checked = validate_with_partial_recovery(data, plan, sources)
    assert checked == {'reviewed': None, 'rejections': [{'reason_code': 'claim_limit_exceeded'}]}


def test_partially_valid_all_facets_still_marked_partial():
    plan, _, sources = _data()
    data = _claims(sources)
    for i, item in enumerate(data['claims']):
        item['claim'] += f' Aspect {chr(65 + i)}.'
    data['claims'].append({'step_id': 'retrieve_01',
                           'claim': 'Unsupported numeric outcome was 75.2%.',
                           'citations': [{'evidence_id': 'E01', 'quote': PHRASE}]})
    checked = validate_with_partial_recovery(data, plan, sources)
    reviewed = checked['reviewed']
    assert reviewed['status'] == 'partial_answer_unverified'
    assert reviewed['rejected_claims_count'] == 1
    assert reviewed['semantic_entailment_verified'] is False
    assert reviewed['human_review_required'] is True
    assert checked['rejections'][0]['reason_code'] == 'numeric_claim_unanchored'


def test_diagnostics_do_not_echo_api_key_or_model_output(graph_shim, tmp_path):
    payload = {'secret_text': 'Never show the raw draft'}
    def fake_llm(_request):
        return {'claims': [{'step_id': 'retrieve_01',
                            'claim': 'This intentionally invalid quote is not in any trusted source.',
                            'citations': [{'evidence_id': 'E01',
                                           'quote': payload['secret_text'] * 3}]}]}
    result = execute(tmp_path, FakeTools(), fake_llm)
    assert result['status'] == 'citation_validation_failed'
    assert result['synthesis']['validation_rejections'][0]['reason_code'] == 'quote_not_exact_in_source'
    assert payload['secret_text'] not in json.dumps(result)
    assert result['llm_calls'] == 1


def test_graph_recovers_only_valid_claims_without_extra_llm_calls(graph_shim, tmp_path):
    calls = []
    def fake_llm(request):
        calls.append(True)
        first, second = request['evidence'][:2]
        return {'claims': [
            {'step_id': first['step_id'], 'claim': 'The source excerpt provides supporting research discussion.',
             'citations': [{'evidence_id': first['evidence_id'], 'quote': PHRASE}]},
            {'step_id': second['step_id'], 'claim': 'An unrelated claim with a fabricated quote.',
             'citations': [{'evidence_id': second['evidence_id'],
                            'quote': 'The fake quote should never be published.'}]},
        ]}
    result = execute(tmp_path, FakeTools(), fake_llm)
    assert result['status'] == 'partial_answer_unverified'
    assert result['answer_generated'] is True
    assert 'fake quote' not in result['answer']
    assert result['synthesis']['rejected_claims_count'] == 1
    assert result['llm_calls'] == 1 and len(calls) == 1


def test_phase3_5_3_graph_accepts_anchor_selections_for_all_facets(graph_shim, tmp_path):
    """End-to-end fake provider: three facets, one local graph run, zero paid calls."""
    calls = []
    def fake_llm(request):
        calls.append(request)
        output = []
        for i, source in enumerate(request['evidence']):
            if source['evidence_id'] not in {'E01', 'E03', 'E05'}:
                continue
            output.append({
                'step_id': source['step_id'],
                'claim': f'The paper discusses the evaluation context for research topic {chr(65+i)}.',
                'citations': [{'evidence_id': source['evidence_id'],
                               'quote_anchor_id': source['quote_options'][0]['quote_anchor_id']}],
            })
        return {'claims': output}

    output = execute(tmp_path, FakeTools(), fake_llm)
    assert output['status'] == 'cited_answer_unverified'
    assert output['synthesis']['claim_count'] == 3
    assert output['synthesis'].get('rejected_claims_count', 0) == 0
    assert output['answer_generated'] is True
    assert output['llm_calls'] == len(calls) == 1
    assert all('quote' in x['citations'][0] for x in output['synthesis']['claims'])
    assert all('quote_anchor_id' not in x['citations'][0] for x in output['synthesis']['claims'])
    assert output['semantic_entailment_verified'] is False

"""Phase 3.5.3: quote IDs eliminate brittle model copying without relaxing provenance."""
import json

from papertrail.agent.synthesis import (
    build_evidence_register, make_synthesis_payload,
    validate_with_partial_recovery, OpenAICompatibleAnswerGenerator,
)

PAPER = '2510.22344'
CITATION = f'https://arxiv.org/abs/{PAPER}#page='


def prepared():
    plan = [{'id': f'retrieve_0{i}', 'proposed_tool': 'hybrid_retrieval',
             'search_query': f'evaluate research facet {i}'} for i in range(1, 4)]
    shared = 'The new method improves performance on complex multi-hop reasoning datasets.'
    metric = 'Evaluation used Exact Match, F1 Score, and judge-based accuracy on multiple benchmarks.'
    limitation = 'The authors report component-level failures and architectural failure modes in their analysis.'
    texts = [(shared, metric), (metric, shared), (limitation,)]
    # The same chunk is indexed into two facets but receives different display IDs.
    ids = [('shared', 'metric'), ('metric', 'shared'), ('limitations',)]
    evidence = []
    for i, step in enumerate(plan):
        hits = [dict(chunk_id=ids[i][j], text=passage,
                     page_number=(18 if ids[i][j] == 'limitations' else 13),
                     citation=f"{CITATION}{18 if ids[i][j] == 'limitations' else 13}", context=[])
                for j, passage in enumerate(texts[i])]
        evidence.append({'step_id': step['id'], 'hits': hits})
    return plan, build_evidence_register(plan, evidence, PAPER)


def anchor_citation(source):
    return {'evidence_id': source['evidence_id'],
            'quote_anchor_id': source['quote_options'][0]['quote_anchor_id']}


def claim(step, source, n):
    endings = ['for method comparison', 'for evaluation metrics', 'for study limitations']
    return {'step_id': step, 'claim': f'The study reports this research facet as supported evidence {endings[n % 3]}. ',
            'citations': [anchor_citation(source)]}


def test_each_quote_anchor_is_a_verbatim_span_in_vetted_passage():
    plan, register = prepared()
    payload = make_synthesis_payload('Compare methods, metrics and limitations.', plan, register)
    assert payload['evidence']
    for row in payload['evidence']:
        for option in row['quote_options']:
            assert option['quote'] in next(r for r in register if r['evidence_id'] == row['evidence_id'])['passages'][0]
            assert len(option['quote']) >= 20
    assert len(json.dumps(payload)) < 18_000


def test_one_claim_per_facet_can_be_selected_without_copying_quotes():
    plan, register = prepared()
    draft = {'claims': [claim(step['id'], next(r for r in register if r['step_id'] == step['id']), i)
                        for i, step in enumerate(plan)]}
    result = validate_with_partial_recovery(draft, plan, register)
    assert not result['rejections']
    assert result['reviewed']['status'] == 'cited_answer_unverified'
    assert result['reviewed']['claim_count'] == 3
    assert result['reviewed']['semantic_entailment_verified'] is False
    for row in result['reviewed']['claims']:
        assert row['citations'][0]['quote']
        assert 'quote_anchor_id' not in row['citations'][0]


def test_fake_anchor_id_fails_closed_and_does_not_leak_text():
    plan, register = prepared()
    draft = {'claims': [claim('retrieve_01', register[0], 1)]}
    draft['claims'][0]['citations'][0]['quote_anchor_id'] = 'FABRICATED_ID'
    result = validate_with_partial_recovery(draft, plan, register)
    assert result['reviewed'] is None
    assert result['rejections'] == [{'claim_index': 1, 'reason_code': 'quote_anchor_id_invalid'}]
    assert 'FABRICATED_ID' not in json.dumps(result)


def test_alias_repair_only_when_exact_same_chunk_registered_for_facet():
    plan, register = prepared()
    # E01 (comparison) and E04 (metrics) index exactly the same raw chunk 'shared'.
    wrong_facet_alias = claim('retrieve_02', register[0], 1)
    accepted = validate_with_partial_recovery({'claims': [wrong_facet_alias]}, plan, register)
    assert accepted['reviewed'] is not None
    assert accepted['reviewed']['claims'][0]['citations'][0]['evidence_id'] == 'E04'
    # Another chunk is not an alias, even if the page happens to be the same.
    other = claim('retrieve_03', register[0], 1)
    rejected = validate_with_partial_recovery({'claims': [other]}, plan, register)
    assert rejected['reviewed'] is None
    assert rejected['rejections'][0]['reason_code'] == 'citation_id_or_facet_invalid'


def test_mixed_valid_and_invalid_anchors_salvaged_with_no_fake_quotes():
    plan, register = prepared()
    valid = claim('retrieve_01', register[0], 1)
    invalid = claim('retrieve_02', register[2], 2)
    invalid['citations'][0]['quote_anchor_id'] = 'E03Q99'
    result = validate_with_partial_recovery({'claims': [valid, invalid]}, plan, register)
    assert result['reviewed']['claim_count'] == 1
    assert result['reviewed']['status'] == 'partial_answer_unverified'
    assert result['rejections'][0] == {'claim_index': 2, 'reason_code': 'quote_anchor_id_invalid'}


def test_legacy_verbatim_quote_style_remains_supported():
    plan, register = prepared()
    draft = {'claims': [{'step_id': 'retrieve_01', 'claim': 'The dataset is an appropriate comparison reference.',
                         'citations': [{'evidence_id': 'E01', 'quote': register[0]['passages'][0]}]}]}
    assert validate_with_partial_recovery(draft, plan, register)['reviewed'] is not None


def test_generator_system_prompt_requests_anchor_ids_not_hand_copied_quotes(monkeypatch):
    captured = []
    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {'choices': [{'message': {'content': '{"claims": []}'}}]}
    class FakeClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, headers, json):
            captured.append(json)
            return FakeResponse()
    monkeypatch.setattr('papertrail.agent.synthesis.httpx.Client', FakeClient)
    provider = OpenAICompatibleAnswerGenerator(api_key='dummy', model='fake', base_url='https://example.org/v1')
    provider(make_synthesis_payload('question', *prepared()))
    assert len(captured) == 1
    sys_prompt = captured[0]['messages'][0]['content']
    assert 'quote_anchor_id' in sys_prompt
    assert 'DO NOT supply quote text' in sys_prompt
    assert captured[0]['temperature'] == 0


def test_long_five_facet_payload_preserves_facets_under_budget():
    plan = [{'id': f'retrieve_{i:02}', 'proposed_tool': 'hybrid_retrieval', 'search_query': 'research facet'}
            for i in range(1, 6)]
    context = 'This is detailed scientific supporting context with careful framing and necessary definitions. ' * 9
    evidence = []
    for i, step in enumerate(plan):
        hits = []
        for j in range(3):
            page = 10+i+j
            hits.append({'chunk_id': f'{step["id"]}-{j}', 'page_number': page,
                         'citation': f'{CITATION}{page}', 'text': context + f' Facet {i}.',
                         'context': [dict(citation=f'{CITATION}{page}', page_number=page,
                                          text=context[::-1]) for _ in range(2)]})
        evidence.append({'step_id': step['id'], 'hits': hits})
    register = build_evidence_register(plan, evidence, PAPER)
    assert len(register) == 10
    payload = make_synthesis_payload('Compare five dimensions of the approaches.', plan, register)
    assert len(json.dumps(payload, ensure_ascii=False)) <= 18_000
    assert {r['step_id'] for r in payload['evidence']} == {p['id'] for p in plan}
    assert all(r['quote_options'] for r in payload['evidence'])


def test_same_chunk_id_at_another_page_cannot_be_remapped():
    plan, register = prepared()
    changed = [dict(r) for r in register]
    alias = next(r for r in changed if r['step_id'] == 'retrieve_02' and r['chunk_id'] == 'shared')
    alias['page_number'] = 100
    alias['citation'] = f'{CITATION}100'
    candidate = claim('retrieve_02', register[0], 0)
    result = validate_with_partial_recovery({'claims': [candidate]}, plan, changed)
    assert result['reviewed'] is None
    assert result['rejections'][0]['reason_code'] == 'citation_id_or_facet_invalid'

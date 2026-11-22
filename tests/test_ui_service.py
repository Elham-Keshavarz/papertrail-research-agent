"""UI contracts: no silent network/API and local-only human decisions."""
import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from papertrail.config import Settings
from papertrail.ui import service


def configured(tmp_path, key=None):
    return Settings(data_dir=tmp_path, OPENAI_API_KEY=key)


def test_local_library_lists_only_valid_artifacts(tmp_path):
    for folder, filename in (
        ('chunks', '2510.22344.json'),
        ('embeddings', '2510.22344.npy'),
        ('embeddings', '2510.22344.manifest.json'),
        ('indexes', '2510.22344.faiss'),
        ('indexes', '2510.22344.manifest.json'),
        ('chunks', 'not-an-arxiv-id.json'),
    ):
        path = tmp_path / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    items = service.list_local_papers(tmp_path)
    assert len(items) == 1
    assert items[0].paper_id == '2510.22344'
    assert items[0].ready is True
    assert service.paper_artifacts('2510.22344', tmp_path).extracted is False


def test_run_preview_uses_real_backend_contract_without_network_or_llm(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(service, 'paper_artifacts', lambda *_: service.LocalPaper('2510.22344', True, True, True, True))

    def fake_research(question, **kwargs):
        seen.update(kwargs)
        return {'status': 'ready_for_llm_synthesis', 'llm_calls': 0}

    monkeypatch.setattr(service, 'run_agent_research', fake_research)
    reported = []
    result = service.run_question(settings=configured(tmp_path), paper_id='2510.22344',
        question='What are the evidence-based limitations of retrieval systems?',
        mode='preview', query_style='budgeted', top_k=5, on_progress=lambda p, s: reported.append((p, s)))
    assert result['llm_calls'] == 0
    assert seen['options'].allow_network is False
    assert seen['options'].query_style == 'budgeted'
    assert seen['options'].top_k == 5
    assert seen['synthesis_provider'] is None
    assert seen['planner_mode'] == 'deterministic'
    assert callable(seen['on_progress'])
    seen['on_progress'](35, 'Local index ready')
    assert reported == [(35, 'Local index ready')]


def test_live_needs_approval_key_and_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'paper_artifacts', lambda *_: service.LocalPaper('2510.22344', True, True, True, True))
    settings = configured(tmp_path, key='TEST_NOT_REAL_KEY')
    question = 'Compare retrieval methods and explain their limitations.'
    with pytest.raises(service.UIServiceError, match='approval'):
        service.run_question(settings=settings, paper_id='2510.22344', question=question,
                             mode='live', live_confirmed=False)
    with pytest.raises(service.UIServiceError, match='Preview-only'):
        service.run_question(settings=settings, paper_id='2510.22344', question=question,
                             mode='live', query_style='focused', live_confirmed=True)
    with pytest.raises(service.UIServiceError, match='OPENAI_API_KEY'):
        service.run_question(settings=configured(tmp_path, key=''), paper_id='2510.22344',
                             question=question, mode='live', live_confirmed=True)


def test_live_approval_wires_real_generator_without_leaking_key(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'paper_artifacts', lambda *_: service.LocalPaper('2510.22344', True, True, True, True))
    providers = []
    class FakeProvider:
        def __init__(self, **kwargs):
            providers.append(kwargs)
    seen = {}
    def fake_research(question, **kwargs):
        seen.update(kwargs)
        return {'status': 'cited_answer_unverified', 'llm_calls': 1}
    monkeypatch.setattr(service, 'OpenAICompatibleAnswerGenerator', FakeProvider)
    monkeypatch.setattr(service, 'run_agent_research', fake_research)
    settings = configured(tmp_path, key='DUMMY_ONLY_NOT_REAL')
    result = service.run_question(settings=settings, paper_id='2510.22344',
        question='Compare retrieval methods and explain their limitations.', mode='live', live_confirmed=True)
    assert result['llm_calls'] == 1
    assert providers[0]['api_key'] == 'DUMMY_ONLY_NOT_REAL'
    assert isinstance(seen['synthesis_provider'], FakeProvider)
    assert seen['options'].allow_network is False
    assert seen['options'].query_style == 'baseline'


def test_discovery_and_import_require_network_consent(tmp_path):
    with pytest.raises(service.UIServiceError, match='network consent'):
        service.discover_papers('retrieval augmented generation', settings=configured(tmp_path), consent=False)
    with pytest.raises(service.UIServiceError, match='consent'):
        service.prepare_paper('2510.22344', settings=configured(tmp_path), consent=False)


def sample_live_result():
    citation = {'evidence_id': 'E01', 'page_number': 4,
                'citation': 'https://arxiv.org/abs/2510.22344#page=4',
                'quote': 'A valid supporting quote that appears in the selected paper.'}
    return {'generation_mode': 'live', 'answer_generated': True,
            'llm_calls': 1,
            'synthesis': {
                'status': 'cited_answer_unverified',
                'claims': [{'step_id': 'retrieve_01', 'claim': 'The study reports a specific approach to retrieval.',
                            'citations': [citation]}],
                'faithfulness_audit': {
                    'status': 'needs_human_semantic_review', 'accepted_claim_count': 1,
                    'claims': [{'claim_index': 1, 'step_id': 'retrieve_01',
                                'evidence_ids': ['E01'], 'mechanical_check': 'quote_anchored_nonverbatim'}],
                },
            }}


def test_human_review_queue_saved_and_updated_locally(tmp_path):
    source = sample_live_result()
    path = service.new_live_review_queue(source, data_dir=tmp_path)
    assert path.is_file()
    assert path.parent == tmp_path / 'faithfulness_reviews'
    assert path in service.available_review_queues(tmp_path)
    report = service.read_review_queue(path, data_dir=tmp_path)
    assert report['review_queue'][0]['reviewer_decision'] == 'pending'
    summary = service.update_review_decision(path, data_dir=tmp_path,
        claim_index=1, decision='insufficient_context', note='Original PDF needs inspection.')
    assert summary['pending'] == 0
    assert summary['insufficient_context'] == 1
    saved = service.read_review_queue(path, data_dir=tmp_path)
    assert saved['review_queue'][0]['reviewer_note'] == 'Original PDF needs inspection.'
    assert summary['semantics_checked_by_software'] is False


def test_review_fails_closed_for_invalid_decisions_and_paths(tmp_path):
    path = service.new_live_review_queue(sample_live_result(), data_dir=tmp_path)
    with pytest.raises(service.UIServiceError, match='decision'):
        service.update_review_decision(path, data_dir=tmp_path,
            claim_index=1, decision='automatically_verified', note='fake')
    with pytest.raises(service.UIServiceError):
        service.update_review_decision(path, data_dir=tmp_path,
            claim_index=2, decision='supported', note='other')
    invalid_path = tmp_path / 'outside_review_queue.json'
    invalid_path.write_text('{}')
    with pytest.raises(service.UIServiceError):
        service.read_review_queue(invalid_path, data_dir=tmp_path)


def test_preview_cannot_create_fake_manual_review_queue(tmp_path):
    value = sample_live_result()
    value['generation_mode'] = 'preview'
    with pytest.raises(service.UIServiceError):
        service.new_live_review_queue(value, data_dir=tmp_path)


def test_ui_script_compiles_and_uses_importable_public_helpers():
    import ast
    source = (Path(__file__).resolve().parents[1] / 'streamlit_app.py').read_text(encoding='utf-8')
    ast.parse(source)
    assert 'run_question(' in source
    assert 'new_live_review_queue(' in source
    assert 'st.form_submit_button' in source


def test_import_progress_only_after_successful_real_stages(tmp_path, monkeypatch):
    import papertrail.agent.tool_execution as te
    states = []
    stages = []
    class FakeTools:
        def ingest(self, *args):
            stages.append('ingest')
        def chunk(self, *args):
            stages.append('chunk')
        def embed(self, *args):
            stages.append('embed')
        def index(self, *args):
            stages.append('index')
    monkeypatch.setattr(service, 'PaperTrailTools', FakeTools)
    def artifacts(*args):
        ready = len(stages) == 4
        return service.LocalPaper('2510.22344', ready, ready, ready, ready)
    monkeypatch.setattr(service, 'paper_artifacts', artifacts)
    value = service.prepare_paper('2510.22344', settings=configured(tmp_path), consent=True,
                                  on_progress=lambda pct, message: states.append((pct, message)))
    assert value.ready
    assert stages == ['ingest', 'chunk', 'embed', 'index']
    assert [p for p, _ in states] == [0, 25, 50, 80, 95, 100]


def test_import_progress_never_claims_completion_after_failure(tmp_path, monkeypatch):
    import papertrail.agent.tool_execution as te
    class FakeTools:
        def ingest(self, *args):
            raise RuntimeError('ingestion failed')
    monkeypatch.setattr(service, 'PaperTrailTools', FakeTools)
    monkeypatch.setattr(service, 'paper_artifacts', lambda *_: service.LocalPaper('2510.22344', False, False, False, False))
    reports = []
    with pytest.raises(RuntimeError):
        service.prepare_paper('2510.22344', settings=configured(tmp_path), consent=True,
                              on_progress=lambda percent, label: reports.append(percent))
    assert reports == [0]

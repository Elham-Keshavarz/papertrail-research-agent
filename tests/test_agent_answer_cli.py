"""Phase 3.5 CLI: key requirements, no surprise network, failure exit codes."""
import json

from papertrail import cli

Q = 'Compare FAIR-RAG and ordinary RAG on evaluation and limitations.'


def _args(*extra):
    return ['agent-answer', '--paper-id', '2510.22344', '--question', Q, *extra]


def test_preview_does_not_need_key_or_send_llm_request(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    def simulate(question, **kwargs):
        assert kwargs['synthesize_mode'] is True
        assert kwargs['synthesis_provider'] is None
        assert kwargs['options'].allow_network is False
        assert kwargs['planner_mode'] == 'deterministic'
        return {'status': 'ready_for_llm_synthesis', 'answer': None, 'evidence': [{'large': 'secret'}],
                'synthesis': {'status': 'ready_for_llm_synthesis'}, 'llm_calls': 0}
    monkeypatch.setattr(cli, 'run_agent_research', simulate)
    assert cli.main(_args()) == 0
    response = json.loads(capsys.readouterr().out)
    assert 'evidence' not in response and response['evidence_content_suppressed_in_cli']
    assert response['llm_calls'] == 0


def test_live_without_key_fails_before_any_tool(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    def no_research(*args, **kwargs):
        raise AssertionError('No tools must run before the API key check')
    monkeypatch.setattr(cli, 'run_agent_research', no_research)
    assert cli.main(_args('--mode', 'live')) == 2
    errors = capsys.readouterr().err
    assert 'OPENAI_API_KEY' in errors and 'Traceback' not in errors


def test_live_mode_passes_validated_model_but_not_key_to_graph_state(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-key-for-unit-test')
    def simulate(question, **kwargs):
        assert kwargs['synthesize_mode'] is True
        provider = kwargs['synthesis_provider']
        assert isinstance(provider, cli.OpenAICompatibleAnswerGenerator)
        assert kwargs['options'].allow_network is False
        return {'status': 'cited_answer_unverified', 'answer': '- A cited claim [E01](https://arxiv.org/abs/2510.22344#page=1)',
                'synthesis': {'status': 'cited_answer_unverified'}, 'llm_calls': 1, 'answer_generated': True}
    monkeypatch.setattr(cli, 'run_agent_research', simulate)
    assert cli.main(_args('--mode', 'live')) == 0
    response = capsys.readouterr().out
    assert 'fake-key-for-unit-test' not in response
    assert json.loads(response)['status'] == 'cited_answer_unverified'


def test_live_failure_exit_code_is_nonzero(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-key-for-unit-test')
    monkeypatch.setattr(cli, 'run_agent_research', lambda *_a, **_k: {'status': 'citation_validation_failed', 'answer': None})
    assert cli.main(_args('--mode', 'live')) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'citation_validation_failed'


def test_unsafe_live_endpoint_cannot_be_used(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-key-for-unit-test')
    monkeypatch.setenv('PAPERTRAIL_OPENAI_BASE_URL', 'http://not-local.example/v1')
    assert cli.main(_args('--mode', 'live')) == 2
    assert 'Validation error' in capsys.readouterr().err


def test_preview_disallows_llm_planner_even_when_key_exists(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-key-for-unit-test')
    monkeypatch.setattr(cli, 'run_agent_research', lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('No model calls')))
    assert cli.main(_args('--mode', 'preview', '--planner', 'llm')) == 2
    assert 'Preview always uses' in capsys.readouterr().err

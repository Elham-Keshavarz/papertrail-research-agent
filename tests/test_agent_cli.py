"""CLI behavior for the new opt-in agent preview."""

import json

from papertrail import cli


def test_parser_registers_preview_command():
    args = cli.build_parser().parse_args(["agent-preview", "--question", "How can a graph coordinate retrieval?"])
    assert args.command == "agent-preview"
    assert args.paper_id is None


def test_preview_cli_displays_json_without_leaking_input_text(monkeypatch, capsys):
    def fake_preview(question, *, paper_id, data_dir, max_papers):
        assert paper_id == "2510.22344"
        assert max_papers == 5
        return {"status": "missing_local_artifacts", "llm_calls": 0, "tool_calls": 0}

    monkeypatch.setattr(cli, "run_agent_preview", fake_preview)
    assert cli.main(["agent-preview", "--question", "How many benchmarks were tested?", "--paper-id", "2510.22344"]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["status"] == "missing_local_artifacts"
    assert rendered["llm_calls"] == 0


def test_preview_cli_invalid_question_is_clean(capsys):
    assert cli.main(["agent-preview", "--question", "short"]) == 2
    captured = capsys.readouterr()
    assert "Validation error:" in captured.err
    assert "Traceback" not in captured.err


def test_preview_cli_invalid_paper_id_is_clean(capsys):
    assert cli.main(["agent-preview", "--question", "How many benchmarks were tested?", "--paper-id", "../evil"]) == 2
    assert "Invalid arXiv" in capsys.readouterr().err


def test_preview_cli_missing_langgraph_reports_actionable_message(monkeypatch, capsys):
    from papertrail.agent import AgentDependencyError

    def unavailable(*args, **kwargs):
        raise AgentDependencyError('LangGraph is missing. Install with: python -m pip install -e ".[agent]"')

    monkeypatch.setattr(cli, "run_agent_preview", unavailable)
    assert cli.main(["agent-preview", "--question", "What does the paper explain?"]) == 2
    output = capsys.readouterr()
    assert "[agent]" in output.err
    assert "Traceback" not in output.err

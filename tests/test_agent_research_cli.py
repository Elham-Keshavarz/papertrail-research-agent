"""CLI flag routing, opt-in and exit-code semantics without network or ML."""
import json

from papertrail import cli


def test_agent_research_cli_uses_explicit_offline_defaults(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    def pretend(question, **kwargs):
        assert kwargs["paper_id"] == "2510.22344"
        assert kwargs["options"].allow_network is False
        assert kwargs["options"].top_k == 2
        assert kwargs["options"].max_tool_calls == 7
        assert kwargs["planner_api_key"] is None
        return {"status": "evidence_retrieved_no_answer", "answer_generated": False, "tool_calls": 2}
    monkeypatch.setattr(cli, "run_agent_research", pretend)
    rc = cli.main(["agent-research", "--paper-id", "2510.22344", "--question",
                   "Which datasets were used to evaluate FAIR-RAG?", "--top-k", "2",
                   "--max-tool-calls", "7"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["tool_calls"] == 2


def test_agent_research_cli_requires_explicit_network(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    def pretend(question, **kwargs):
        options = kwargs["options"]
        assert options.allow_network and options.auto_select_first
        assert kwargs["planner_api_key"] is None
        return {"status": "paper_selection_required", "tool_calls": 1}
    monkeypatch.setattr(cli, "run_agent_research", pretend)
    rc = cli.main(["agent-research", "--question", "Which methods compare to the FAIR-RAG baseline?",
                   "--allow-network", "--auto-select-first"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["status"] == "paper_selection_required"


def test_agent_research_cli_invalid_flag_is_clean_error(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["agent-research", "--question", "Which datasets did FAIR-RAG evaluate?",
                   "--auto-select-first"])
    assert rc == 2
    assert "Validation error" in capsys.readouterr().err


def test_agent_research_cli_failure_nonzero(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_agent_research",
        lambda *args, **kwargs: {"status": "tool_failed", "errors": ["Index verification failed"]})
    rc = cli.main(["agent-research", "--paper-id", "2510.22344",
                   "--question", "Which datasets were used to evaluate FAIR-RAG?"])
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["status"] == "tool_failed"

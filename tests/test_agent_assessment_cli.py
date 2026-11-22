"""Phase 3.4 CLI opt-in/key handling and clean exit codes."""
import json

from papertrail import cli


def test_agent_assess_cli_offline_defaults(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)

    def pretend(question, **kwargs):
        assert question == "What evidence supports FAIR-RAG evaluation results?"
        assert kwargs["paper_id"] == "2510.22344"
        assert kwargs["assess_mode"] is True
        assert kwargs["options"].allow_network is False
        assert kwargs["options"].max_refinement_rounds == 1
        assert kwargs["options"].min_hits_per_step == 2
        assert kwargs["planner_api_key"] is None
        return {"status": "evidence_assessed_unverified", "assessment": {"decision": "structurally_covered_unverified"}}

    monkeypatch.setattr(cli, "run_agent_research", pretend)
    code = cli.main(["agent-assess", "--paper-id", "2510.22344", "--question",
                     "What evidence supports FAIR-RAG evaluation results?"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "evidence_assessed_unverified"


def test_agent_assess_rejects_unbounded_configuration(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    code = cli.main(["agent-assess", "--paper-id", "2510.22344", "--question",
                     "What evidence supports FAIR-RAG evaluation results?",
                     "--max-refinement-rounds", "22"])
    assert code == 2
    assert "Validation error" in capsys.readouterr().err


def test_agent_assess_returns_zero_for_audited_gaps(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_agent_research",
                        lambda *_a, **_kw: {"status": "evidence_gaps_detected", "answer_generated": False})
    code = cli.main(["agent-assess", "--paper-id", "2510.22344", "--question",
                     "What evidence supports FAIR-RAG evaluation results?", "--max-refinement-rounds", "0"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "evidence_gaps_detected"


def test_agent_assess_returns_nonzero_for_failed_tool(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_agent_research",
                        lambda *_a, **_kw: {"status": "tool_failed", "errors": ["TimeoutError"]})
    code = cli.main(["agent-assess", "--paper-id", "2510.22344", "--question",
                     "What evidence supports FAIR-RAG evaluation results?"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "tool_failed"

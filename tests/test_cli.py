import json

from papertrail.cli import main


def test_doctor_cli_without_api_key(monkeypatch, capsys):
    # An explicit empty environment value overrides the user's .env file.
    # This keeps the test independent of whether a developer has configured a key.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert main(["doctor"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["phase"] == 6
    assert result["milestone"] == "6.0"
    assert result["arxiv_discovery_available"] is True
    assert result["pdf_ingestion_available"] is True
    assert result["chunking_available"] is True
    assert result["embedding_available"] is True
    assert result["api_key_configured"] is False
    assert result["live_research_available"] is False


def test_doctor_cli_with_api_key_env(monkeypatch, capsys):
    # Synthetic credentials must never be used to contact a real provider.
    marker = "DOCTOR_TEST_FAKE_KEY_NEVER_SEND"
    monkeypatch.setenv("OPENAI_API_KEY", marker)
    assert main(["doctor"]) == 0
    stdout = capsys.readouterr().out
    result = json.loads(stdout)
    assert result["api_key_configured"] is True
    assert result["live_research_available"] is result["langgraph_backend_installed"]
    assert marker not in stdout


def test_doctor_cli_with_configured_key_and_langgraph_backend(monkeypatch, capsys):
    # Test the exact Windows condition, even if LangGraph is absent on CI.
    from papertrail import health

    marker = "DOCTOR_TEST_CONFIGURED_BACKEND_FAKE_KEY"
    monkeypatch.setenv("OPENAI_API_KEY", marker)
    original_find_spec = health.find_spec
    monkeypatch.setattr(
        health,
        "find_spec",
        lambda name: object() if name == "langgraph" else original_find_spec(name),
    )
    assert main(["doctor"]) == 0
    stdout = capsys.readouterr().out
    result = json.loads(stdout)
    assert result["langgraph_backend_installed"] is True
    assert result["api_key_configured"] is True
    assert result["live_research_available"] is True
    assert marker not in stdout


def test_doctor_cli_with_dotenv_key(tmp_path, monkeypatch, capsys):
    # Regression: copying .env into a release previously broke test_doctor_cli.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    marker = "DOCTOR_TEST_DOTENV_FAKE_KEY"
    (tmp_path / ".env").write_text(f"OPENAI_API_KEY={marker}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["doctor"]) == 0
    stdout = capsys.readouterr().out
    result = json.loads(stdout)
    assert result["api_key_configured"] is True
    assert result["live_research_available"] is result["langgraph_backend_installed"]
    assert marker not in stdout


def test_validate_question_cli(capsys):
    assert main(["validate-question", "--question", "How can RAG reduce hallucinations?"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "valid"
    assert result["research_executed"] is False


def test_short_question_fails_without_traceback(capsys):
    assert main(["validate-question", "--question", "too short"]) == 2
    output = capsys.readouterr()
    assert "Validation error" in output.err
    assert "Traceback" not in output.err


def test_search_cli_renders_json_without_real_network(monkeypatch, capsys):
    from papertrail import cli
    from papertrail.discovery.arxiv_parser import parse_arxiv_feed
    from pathlib import Path

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def search(self, query, **kwargs):
            assert query == "retrieval augmented generation"
            assert kwargs["max_results"] == 2
            return parse_arxiv_feed(
                (Path(__file__).parent / "fixtures" / "arxiv_sample.xml").read_bytes(),
                query=query, start=0, requested=2,
            )

    monkeypatch.setattr(cli, "ArxivClient", FakeClient)
    assert main(["search-papers", "--query", "retrieval augmented generation", "--limit", "2"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["papers"][0]["paper_id"] == "2401.12345"
    assert len(output["papers"]) == 2


def test_search_cli_handles_invalid_bounds(capsys):
    assert main(["search-papers", "--query", "retrieval", "--limit", "-3"]) == 2
    assert "Validation error" in capsys.readouterr().err


def test_search_cli_handles_upstream_failure(monkeypatch, capsys):
    from papertrail import cli
    from papertrail.discovery import ArxivRequestError

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def search(self, *args, **kwargs):
            raise ArxivRequestError("fake unavailable")

    monkeypatch.setattr(cli, "ArxivClient", FakeClient)
    assert main(["search-papers", "--query", "retrieval"]) == 1
    assert "arXiv error" in capsys.readouterr().err


def test_zero_limit_is_not_silently_replaced(capsys):
    assert main(["search-papers", "--query", "retrieval", "--limit", "0"]) == 2
    assert "Validation error" in capsys.readouterr().err


def test_ingest_pdf_cli_offline(monkeypatch, tmp_path, capsys):
    import pymupdf
    from papertrail import cli

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "A page of research findings")
    fixture_pdf = doc.tobytes()
    doc.close()

    class FakeDownloader:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def fetch(self, paper_id):
            return fixture_pdf

    monkeypatch.setattr(cli, "ArxivPdfDownloader", FakeDownloader)
    monkeypatch.chdir(tmp_path)
    assert main(["ingest-pdf", "--paper-id", "2510.22344"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "extracted"
    assert data["total_pages"] == 1
    assert data["characters_extracted"] > 0
    assert "research findings" not in json.dumps(data)
    assert (tmp_path / data["extracted_json_path"]).exists()


def test_ingest_pdf_cli_rejects_invalid_id(capsys):
    assert main(["ingest-pdf", "--paper-id", "../../evil"]) == 2
    assert "Invalid arXiv" in capsys.readouterr().err


def test_ingest_pdf_cli_rejects_invalid_page_limit(capsys):
    assert main(["ingest-pdf", "--paper-id", "2510.22344", "--max-pages", "0"]) == 2
    assert "max_pages" in capsys.readouterr().err

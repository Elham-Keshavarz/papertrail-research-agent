"""CLI protects output from raw paper text and handles errors without tracebacks."""
import json
from pathlib import Path

from papertrail import cli
from papertrail.embedding.pipeline import EmbeddingError, EmbeddingManifest, EmbeddingRun


def test_embedding_cli_prints_summary_without_paper_text(monkeypatch, capsys, tmp_path):
    def fake_embed(paper_id, *, data_dir, batch_size, force):
        assert paper_id == "2510.22344" and batch_size == 8 and force
        assert data_dir == Path("data")
        return EmbeddingRun(
            manifest=EmbeddingManifest(
                paper_id=paper_id, source_sha256="a" * 64, chunk_file_sha256="b" * 64,
                model_name="mock-model", model_revision="revision",
                normalized=True, embedding_dimension=384, max_sequence_length=256,
                min_input_tokens=4, max_input_tokens=110,
                vector_count=2, vector_file_sha256="c" * 64,
                chunk_ids=["SECRET PAPER 1", "SECRET PAPER 2"],
            ),
            vectors_path=Path("data/embeddings/test.npy"),
            manifest_path=Path("data/embeddings/test.manifest.json"),
            cached=False,
        )
    monkeypatch.setattr(cli, "embed_saved_document", fake_embed)
    assert cli.main(["embed-document", "--paper-id", "2510.22344", "--batch-size", "8", "--force"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total_vectors"] == 2
    assert result["embedding_dimension"] == 384
    assert "SECRET PAPER" not in json.dumps(result)


def test_embedding_cli_validation_error_no_traceback(capsys):
    assert cli.main(["embed-document", "--paper-id", "2510.22344", "--batch-size", "0"]) == 2
    out = capsys.readouterr()
    assert "batch_size" in out.err and "Traceback" not in out.err


def test_embedding_cli_missing_input_error(capsys, monkeypatch, tmp_path):
    """Do not depend on whether a user has populated the project data/ cache."""
    monkeypatch.chdir(tmp_path)
    assert cli.main(["embed-document", "--paper-id", "2510.22344"]) == 2
    assert "chunk-document first" in capsys.readouterr().err


def test_embedding_cli_model_runtime_failure_is_handled(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise EmbeddingError("Model unavailable")
    monkeypatch.setattr(cli, "embed_saved_document", fail)
    assert cli.main(["embed-document", "--paper-id", "2510.22344"]) == 2
    assert "Model unavailable" in capsys.readouterr().err

"""Chunking from local JSON requires no network, model or downloadable corpus."""

import json

import pytest

from papertrail.chunking.chunker import ChunkingError
from papertrail.chunking.pipeline import chunk_saved_document
from papertrail.ingestion.pdf_extractor import ExtractedDocument, ExtractedPage


def save_extracted(base, text="Research evidence " * 60, paper_id="2510.22344"):
    doc = ExtractedDocument(
        paper_id=paper_id,
        source_url=f"https://arxiv.org/abs/{paper_id}",
        sha256="b" * 64,
        total_pages=1,
        extracted_pages=1,
        truncated=False,
        empty_text_pages=[],
        pages=[ExtractedPage(page_number=1, text=text, character_count=len(text), has_text=bool(text))],
    )
    p = base / "extracted" / f"{paper_id.replace('/', '__')}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc.model_dump_json(), encoding="utf-8")
    return p


def test_chunk_pipeline_reads_existing_json_and_writes_chunks(tmp_path):
    save_extracted(tmp_path)
    chunks, out = chunk_saved_document("2510.22344", data_dir=tmp_path, chunk_size=180, overlap=35)
    assert out == tmp_path / "chunks" / "2510.22344.json"
    content = json.loads(out.read_text(encoding="utf-8"))
    assert content["source_sha256"] == "b" * 64
    assert content["total_chunks"] == chunks.total_chunks
    assert content["chunks"][0]["page_number"] == 1
    assert "Research evidence" in content["chunks"][0]["text"]


def test_pipeline_replay_produces_same_bytes(tmp_path):
    save_extracted(tmp_path)
    _, out = chunk_saved_document("2510.22344", data_dir=tmp_path)
    a = out.read_bytes()
    chunk_saved_document("2510.22344", data_dir=tmp_path)
    assert out.read_bytes() == a


def test_missing_input_has_helpful_message(tmp_path):
    with pytest.raises(ChunkingError, match="Run ingest-pdf first"):
        chunk_saved_document("2510.22344", data_dir=tmp_path)
    assert not (tmp_path / "chunks").exists()


def test_bad_input_json_is_refused_and_does_not_write_output(tmp_path):
    p = save_extracted(tmp_path)
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(ChunkingError, match="invalid"):
        chunk_saved_document("2510.22344", data_dir=tmp_path)
    assert not (tmp_path / "chunks").exists()


def test_mismatch_id_refused(tmp_path):
    p = save_extracted(tmp_path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    obj["paper_id"] = "2502.00306"
    p.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ChunkingError, match="does not match"):
        chunk_saved_document("2510.22344", data_dir=tmp_path)


def test_invalid_id_cannot_read_arbitrary_files(tmp_path):
    with pytest.raises(ValueError):
        chunk_saved_document("../../private", data_dir=tmp_path)


def test_invalid_overlapping_settings_do_not_create_output(tmp_path):
    save_extracted(tmp_path)
    with pytest.raises(ChunkingError, match="overlap"):
        chunk_saved_document("2510.22344", data_dir=tmp_path, chunk_size=120, overlap=70)
    assert not (tmp_path / "chunks").exists()


def test_old_style_id_safe_file_name(tmp_path):
    save_extracted(tmp_path, paper_id="hep-th/9901001")
    _, path = chunk_saved_document("hep-th/9901001", data_dir=tmp_path)
    assert path.name == "hep-th__9901001.json"


def test_no_raw_text_printed_by_cli(monkeypatch, tmp_path, capsys):
    from papertrail.cli import main
    save_extracted(tmp_path / "data", text="SECRET CHUNK DETAILS " * 80)
    monkeypatch.chdir(tmp_path)
    assert main(["chunk-document", "--paper-id", "2510.22344", "--chunk-size", "160", "--overlap", "25"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "chunked"
    assert output["total_chunks"] > 1
    assert output["chunks_json_path"].endswith("2510.22344.json")
    assert "SECRET CHUNK DETAILS" not in json.dumps(output)


def test_cli_missing_json_returns_error_no_traceback(monkeypatch, tmp_path, capsys):
    from papertrail.cli import main
    monkeypatch.chdir(tmp_path)
    assert main(["chunk-document", "--paper-id", "2510.22344"]) == 2
    error = capsys.readouterr().err
    assert "ingest-pdf first" in error
    assert "Traceback" not in error


def test_cli_invalid_settings_no_traceback(monkeypatch, tmp_path, capsys):
    from papertrail.cli import main
    save_extracted(tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    assert main(["chunk-document", "--paper-id", "2510.22344", "--chunk-size", "99"]) == 2
    assert "chunk_size" in capsys.readouterr().err


def test_ingestion_to_chunking_end_to_end_offline(tmp_path):
    """Exercise real PDF -> extracted JSON -> chunk JSON, with a stubbed download."""
    import pymupdf
    from papertrail.ingestion.pipeline import ingest_arxiv_pdf

    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Evidence in first page: research methods and results.")
    pdf.new_page().insert_text((72, 72), "Evidence in second page: evaluation and limitations.")
    payload = pdf.tobytes()
    pdf.close()

    class StubDownloader:
        def fetch(self, paper_id):
            assert paper_id == "2510.22344"
            return payload

    extracted, _, _, _ = ingest_arxiv_pdf(
        "2510.22344", data_dir=tmp_path, client=StubDownloader()
    )
    index, out = chunk_saved_document("2510.22344", data_dir=tmp_path)
    assert index.total_pages == 2
    assert index.total_chunks == 2
    assert index.source_sha256 == extracted.sha256
    assert [chunk.page_number for chunk in index.chunks] == [1, 2]
    for chunk in index.chunks:
        page = extracted.pages[chunk.page_number - 1]
        assert page.text[chunk.start_char:chunk.end_char] == chunk.text
    assert out.exists()


def test_invalid_source_does_not_overwrite_prior_chunk_output(tmp_path):
    path = save_extracted(tmp_path)
    _, output_path = chunk_saved_document("2510.22344", data_dir=tmp_path)
    previous_content = output_path.read_bytes()
    path.write_text("{invalid json", encoding="utf-8")
    with pytest.raises(ChunkingError):
        chunk_saved_document("2510.22344", data_dir=tmp_path)
    assert output_path.read_bytes() == previous_content


def test_extracted_input_size_limit(tmp_path):
    from papertrail.chunking.pipeline import MAX_EXTRACTED_JSON_BYTES

    path = tmp_path / "extracted" / "2510.22344.json"
    path.parent.mkdir(parents=True)
    with path.open("wb") as stream:
        stream.truncate(MAX_EXTRACTED_JSON_BYTES + 1)
    with pytest.raises(ChunkingError, match="size"):
        chunk_saved_document("2510.22344", data_dir=tmp_path)

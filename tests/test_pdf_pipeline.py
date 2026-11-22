import json

import pymupdf
import pytest

from papertrail.ingestion.pdf_downloader import PdfDownloadError
from papertrail.ingestion.pipeline import ingest_arxiv_pdf


@pytest.fixture
def pdf_bytes():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "The reproducible research evidence")
    pdf = doc.tobytes()
    doc.close()
    return pdf


class StubClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def fetch(self, paper_id):
        self.calls.append(paper_id)
        return self.payload


def test_pipeline_saves_pdf_and_page_level_json(tmp_path, pdf_bytes):
    client = StubClient(pdf_bytes)
    document, pdf_path, json_path, cached = ingest_arxiv_pdf("2510.22344", data_dir=tmp_path, client=client)
    assert not cached
    assert pdf_path.read_bytes() == pdf_bytes
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    assert obj["pages"][0]["page_number"] == 1
    assert "reproducible" in obj["pages"][0]["text"]
    assert document.sha256 == obj["sha256"]
    assert client.calls == ["2510.22344"]


def test_second_ingestion_uses_cached_pdf(tmp_path, pdf_bytes):
    client = StubClient(pdf_bytes)
    ingest_arxiv_pdf("2510.22344", data_dir=tmp_path, client=client)
    _, _, _, cached = ingest_arxiv_pdf("2510.22344", data_dir=tmp_path, client=client)
    assert cached is True
    assert client.calls == ["2510.22344"]


def test_force_reloads_pdf(tmp_path, pdf_bytes):
    client = StubClient(pdf_bytes)
    ingest_arxiv_pdf("2510.22344", data_dir=tmp_path, client=client)
    _, _, _, cached = ingest_arxiv_pdf("2510.22344", data_dir=tmp_path, client=client, force=True)
    assert cached is False
    assert client.calls == ["2510.22344", "2510.22344"]


def test_invalid_pdf_does_not_leave_output(tmp_path):
    client = StubClient(b"%PDF-not-really-valid")
    with pytest.raises(Exception):
        ingest_arxiv_pdf("2510.22344", data_dir=tmp_path, client=client)
    assert not list(tmp_path.rglob("*.pdf"))
    assert not list(tmp_path.rglob("*.json"))


def test_old_id_is_safely_normalized_for_filename(tmp_path, pdf_bytes):
    client = StubClient(pdf_bytes)
    _, pdf_path, _, _ = ingest_arxiv_pdf("hep-th/9901001", data_dir=tmp_path, client=client)
    assert pdf_path.name == "hep-th__9901001.pdf"
    assert pdf_path.parent == tmp_path / "papers"


def test_invalid_id_refuses_before_network_or_file_access(tmp_path, pdf_bytes):
    client = StubClient(pdf_bytes)
    with pytest.raises(ValueError):
        ingest_arxiv_pdf("../../secret", data_dir=tmp_path, client=client)
    assert client.calls == []
    assert not list(tmp_path.rglob("*"))


def test_rejects_oversized_cached_file(tmp_path, pdf_bytes):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    (pdf_dir / "2510.22344.pdf").write_bytes(b"x" * (20 * 1024 * 1024 + 1))
    with pytest.raises(PdfDownloadError, match="Cached PDF"):
        ingest_arxiv_pdf("2510.22344", data_dir=tmp_path, client=StubClient(pdf_bytes))


def test_cached_pdf_can_be_reextracted_with_a_different_layout(tmp_path, pdf_bytes):
    client = StubClient(pdf_bytes)
    ingest_arxiv_pdf("2510.22344", data_dir=tmp_path, client=client, layout="simple")
    document, _, output, cached = ingest_arxiv_pdf("2510.22344", data_dir=tmp_path, client=client, layout="auto")
    assert cached
    assert client.calls == ["2510.22344"]
    assert document.pages[0].extraction_method == "simple"
    assert json.loads(output.read_text(encoding="utf-8"))["pages"][0]["extraction_method"] == "simple"


def test_invalid_layout_rejected_before_downloading(tmp_path, pdf_bytes):
    client = StubClient(pdf_bytes)
    with pytest.raises(ValueError, match="layout"):
        ingest_arxiv_pdf("2510.22344", data_dir=tmp_path, client=client, layout="invalid")
    assert client.calls == []

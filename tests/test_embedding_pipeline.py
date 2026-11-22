"""Use fake local models: tests are deterministic, do not download weights, and use no API."""
import io
import json

import numpy as np
import pytest

from papertrail.chunking.pipeline import chunk_saved_document
from papertrail.embedding.pipeline import EmbeddingError, embed_saved_document
from papertrail.embedding.encoder import MODEL_NAME, MODEL_REVISION
from papertrail.ingestion.pdf_extractor import ExtractedDocument, ExtractedPage

PAPER_ID = "2510.22344"


class FakeEncoder:
    dimension = 4
    max_sequence_length = 256
    init_count = 0
    encode_count = 0

    def __init__(self, model_name, revision):
        assert model_name == MODEL_NAME
        assert revision == MODEL_REVISION
        FakeEncoder.init_count += 1

    def token_count(self, text):
        return len(text.split()) + 2

    def encode_documents(self, texts, *, batch_size):
        assert 1 <= batch_size <= 128
        FakeEncoder.encode_count += 1
        return np.array([[len(s), sum(map(ord, s)) % 31 + 1, 2, 1] for s in texts], dtype=np.float32)


@pytest.fixture
def saved_chunks(tmp_path):
    text = ("Evidence-based research uses transparent methods and citations. " * 15)
    doc = ExtractedDocument(
        paper_id=PAPER_ID, source_url=f"https://arxiv.org/abs/{PAPER_ID}",
        sha256="a" * 64, total_pages=1, extracted_pages=1,
        truncated=False, empty_text_pages=[],
        pages=[ExtractedPage(page_number=1, text=text, character_count=len(text), has_text=True)],
    )
    path = tmp_path / "extracted" / f"{PAPER_ID}.json"
    path.parent.mkdir(parents=True)
    path.write_text(doc.model_dump_json(), encoding="utf-8")
    _, output = chunk_saved_document(PAPER_ID, data_dir=tmp_path, chunk_size=180, overlap=25)
    assert output.exists()
    return tmp_path, output


def embed(data_dir, **kwargs):
    return embed_saved_document(PAPER_ID, data_dir=data_dir, encoder_factory=FakeEncoder, **kwargs)


def test_first_run_saves_manifest_and_normalized_matrix(saved_chunks):
    folder, chunks = saved_chunks
    run = embed(folder)
    assert not run.cached
    m = run.manifest
    assert m.vector_count == len(json.loads(chunks.read_text())["chunks"])
    assert m.embedding_dimension == 4
    assert m.model_revision == MODEL_REVISION
    assert m.normalized and m.max_input_tokens <= 256
    matrix = np.load(run.vectors_path, allow_pickle=False)
    assert matrix.dtype == np.float32
    assert matrix.shape == (m.vector_count, 4)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-6)
    assert json.loads(run.manifest_path.read_text())["chunk_ids"] == [c["chunk_id"] for c in json.loads(chunks.read_text())["chunks"]]


def test_second_run_hits_cache_without_loading_model(saved_chunks):
    folder, _ = saved_chunks
    first = embed(folder)
    def impossible_factory(*args):
        raise AssertionError("Model must not be initialized for cache hit")
    second = embed_saved_document(PAPER_ID, data_dir=folder, encoder_factory=impossible_factory)
    assert second.cached and first.manifest == second.manifest


def test_force_reembeds(saved_chunks):
    folder, _ = saved_chunks
    embed(folder)
    initial = FakeEncoder.encode_count
    again = embed(folder, force=True)
    assert not again.cached
    assert FakeEncoder.encode_count == initial + 1


def test_changes_to_chunks_invalidate_cache(saved_chunks):
    folder, file = saved_chunks
    embed(folder)
    # Changes to source bytes cause re-embedding, even when IDs are unchanged.
    source = json.loads(file.read_text())
    source["chunks"][0]["text"] = "Different data " + source["chunks"][0]["text"][15:]
    file.write_text(json.dumps(source))
    first = embed(folder)
    assert not first.cached


def test_corrupt_vector_blob_causes_rebuild(saved_chunks):
    folder, _ = saved_chunks
    run = embed(folder)
    run.vectors_path.write_bytes(b"corrupt")
    assert not embed(folder).cached


def test_corrupt_manifest_causes_rebuild(saved_chunks):
    folder, _ = saved_chunks
    run = embed(folder)
    run.manifest_path.write_text("not json")
    assert not embed(folder).cached


def test_missing_vectors_causes_rebuild(saved_chunks):
    folder, _ = saved_chunks
    run = embed(folder)
    run.vectors_path.unlink()
    assert not embed(folder).cached


def test_missing_chunks_has_actionable_message(tmp_path):
    with pytest.raises(EmbeddingError, match="Run chunk-document first"):
        embed(tmp_path)


@pytest.mark.parametrize("batch", [0, -1, 129, True])
def test_invalid_batch_size_rejected_before_loading_model(tmp_path, batch):
    with pytest.raises(EmbeddingError, match="batch_size"):
        embed(tmp_path, batch_size=batch)


def test_invalid_paper_id_rejected(tmp_path):
    with pytest.raises(ValueError):
        embed_saved_document("../../secret", data_dir=tmp_path)


def test_invalid_chunks_json_rejected(saved_chunks):
    folder, file = saved_chunks
    file.write_text("invalid")
    with pytest.raises(EmbeddingError, match="validate chunk JSON"):
        embed(folder)


def test_mismatch_id_rejected(saved_chunks):
    folder, file = saved_chunks
    data = json.loads(file.read_text())
    data["paper_id"] = "2401.12345"
    file.write_text(json.dumps(data))
    with pytest.raises(EmbeddingError, match="ID does not match"):
        embed(folder)


def test_duplicate_chunk_id_rejected(saved_chunks):
    folder, file = saved_chunks
    data = json.loads(file.read_text())
    data["chunks"][1]["chunk_id"] = data["chunks"][0]["chunk_id"]
    file.write_text(json.dumps(data))
    with pytest.raises(EmbeddingError, match="Duplicate chunk IDs"):
        embed(folder)


def test_empty_chunks_rejected(saved_chunks):
    folder, file = saved_chunks
    data = json.loads(file.read_text())
    data["total_chunks"] = 0
    data["chunks"] = []
    file.write_text(json.dumps(data))
    with pytest.raises(EmbeddingError, match="nonempty"):
        embed(folder)


def test_token_overflow_is_detected_before_weak_model_encode(saved_chunks):
    folder, _ = saved_chunks
    class ShortLimit(FakeEncoder):
        max_sequence_length = 12
        def encode_documents(self, texts, *, batch_size):
            raise AssertionError("Must never encode over-length text")
    with pytest.raises(EmbeddingError, match="No text was silently truncated") as error:
        embed_saved_document(PAPER_ID, data_dir=folder, encoder_factory=ShortLimit)
    assert "--chunk-size 650" in str(error.value)
    assert not (folder / "embeddings").exists()


@pytest.mark.parametrize("bad", [np.array([[float('nan')]*4]), np.zeros((1, 4)), np.ones((1, 2))])
def test_reject_invalid_matrix(saved_chunks, bad):
    folder, _ = saved_chunks
    class Bad(FakeEncoder):
        def encode_documents(self, texts, *, batch_size):
            return bad
    with pytest.raises(EmbeddingError, match="invalid|zero"):
        embed_saved_document(PAPER_ID, data_dir=folder, encoder_factory=Bad)
    assert not (folder / "embeddings" / f"{PAPER_ID}.manifest.json").exists()


def test_revision_is_part_of_cache_key(saved_chunks):
    folder, _ = saved_chunks
    embed(folder)
    class Other(FakeEncoder):
        def __init__(self, model_name, revision):
            assert revision == "revision2"
    result = embed_saved_document(PAPER_ID, data_dir=folder, encoder_factory=Other, model_revision="revision2")
    assert not result.cached and result.manifest.model_revision == "revision2"


def test_cache_does_not_require_sentence_transformers_installed(saved_chunks):
    folder, _ = saved_chunks
    embed(folder)
    # The dependency is lazily imported only when a fresh model is required.
    result = embed_saved_document(PAPER_ID, data_dir=folder)
    assert result.cached


def test_normalized_cache_payload_is_checked_even_with_matching_sha(saved_chunks):
    import hashlib
    folder, _ = saved_chunks
    run = embed(folder)
    v = np.load(run.vectors_path, allow_pickle=False)
    v[0] = v[0] * 3
    buff = io.BytesIO()
    np.save(buff, v, allow_pickle=False)
    blob = buff.getvalue()
    run.vectors_path.write_bytes(blob)
    manifest = json.loads(run.manifest_path.read_text())
    manifest["vector_file_sha256"] = hashlib.sha256(blob).hexdigest()
    run.manifest_path.write_text(json.dumps(manifest))
    assert not embed(folder).cached


def test_end_to_end_pdf_chunk_to_embeddings_with_fake_encoder(tmp_path):
    import pymupdf
    from papertrail.ingestion.pipeline import ingest_arxiv_pdf
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72,72), "Grounded AI answers need reliable sources and citations.")
    pdf.new_page().insert_text((72,72), "Retrieval-based pipelines compare relevant evidence.")
    binary = pdf.tobytes()
    pdf.close()

    class StubDownloader:
        def fetch(self, paper_id):
            assert paper_id == PAPER_ID
            return binary

    doc, _, _, _ = ingest_arxiv_pdf(PAPER_ID, data_dir=tmp_path, client=StubDownloader())
    chunks, _ = chunk_saved_document(PAPER_ID, data_dir=tmp_path)
    result = embed(tmp_path)
    assert result.manifest.vector_count == chunks.total_chunks == doc.extracted_pages == 2
    assert len(result.manifest.chunk_ids) == 2

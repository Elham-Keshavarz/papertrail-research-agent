"""Deterministic, network-free regression suite for vector indexing and citations."""
import hashlib
import io
import json

import numpy as np
import pytest

from papertrail.chunking.pipeline import chunk_saved_document
from papertrail.embedding.encoder import MODEL_NAME, MODEL_REVISION
from papertrail.embedding.pipeline import embed_saved_document
from papertrail.ingestion.pdf_extractor import ExtractedDocument, ExtractedPage
from papertrail.retrieval.faiss_store import RetrievalError, build_paper_index
from papertrail.retrieval.search import search_paper

PAPER = "2510.22344"


class FakeIndexFlatIP:
    metric_type = 0

    def __init__(self, dimension):
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vectors):
        assert vectors.dtype == np.float32
        self.vectors = np.concatenate([self.vectors, vectors])

    def search(self, queries, k):
        similarities = queries @ self.vectors.T
        # Stable tie-break by original position for reproducibility in unit tests.
        order = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(similarities, order, axis=1), order.astype(np.int64)


class FakeFaiss:
    METRIC_INNER_PRODUCT = 0
    IndexFlatIP = FakeIndexFlatIP
    build_count = 0

    @classmethod
    def serialize_index(cls, index):
        cls.build_count += 1
        buffer = io.BytesIO()
        np.save(buffer, index.vectors, allow_pickle=False)
        return np.frombuffer(buffer.getvalue(), dtype=np.uint8)

    @staticmethod
    def deserialize_index(blob):
        array = np.load(io.BytesIO(blob.tobytes()), allow_pickle=False)
        index = FakeIndexFlatIP(array.shape[1])
        index.add(array)
        return index


class FakeEncoder:
    dimension = 4
    max_sequence_length = 256

    def __init__(self, name, revision):
        assert (name, revision) == (MODEL_NAME, MODEL_REVISION)

    def token_count(self, text):
        return len(text.split()) + 2

    @staticmethod
    def _vector(text):
        text = text.lower()
        return np.array([
            text.count("evidence") + text.count("agent"),
            text.count("cat") + text.count("image"),
            0.5, 0.1,
        ], dtype=np.float32)

    def encode_documents(self, texts, *, batch_size):
        return np.stack([self._vector(t) for t in texts])

    def encode_query(self, text):
        return self._vector(text)


@pytest.fixture
def saved(tmp_path):
    pages = [
        "Agent evidence assessment improves grounded retrieval. " * 10,
        "Cat image classification uses visual features and labels. " * 10,
    ]
    doc = ExtractedDocument(
        paper_id=PAPER, source_url=f"https://arxiv.org/abs/{PAPER}", sha256="b" * 64,
        total_pages=2, extracted_pages=2, truncated=False, empty_text_pages=[],
        pages=[ExtractedPage(page_number=i, text=t, character_count=len(t), has_text=True)
               for i, t in enumerate(pages, 1)],
    )
    extracted = tmp_path / "extracted" / f"{PAPER}.json"
    extracted.parent.mkdir()
    extracted.write_text(doc.model_dump_json(), encoding="utf-8")
    chunk_saved_document(PAPER, data_dir=tmp_path, chunk_size=160, overlap=25)
    embed_saved_document(PAPER, data_dir=tmp_path, encoder_factory=FakeEncoder)
    return tmp_path


def index(saved, **kwargs):
    return build_paper_index(PAPER, data_dir=saved, faiss_backend=FakeFaiss, **kwargs)


def search(saved, query="agent evidence", **kwargs):
    return search_paper(PAPER, query, data_dir=saved,
                        encoder_factory=FakeEncoder, faiss_backend=FakeFaiss, **kwargs)


def test_index_creation_and_manifest(saved):
    run = index(saved)
    assert not run.cached
    assert run.manifest.dimension == 4
    assert run.manifest.metric == "inner_product"
    assert run.manifest.index_type == "IndexFlatIP"
    assert run.manifest.vector_count > 2
    assert run.index_path.is_file() and run.manifest_path.is_file()
    assert run.manifest.index_file_sha256 == hashlib.sha256(run.index_path.read_bytes()).hexdigest()


def test_second_build_uses_disk_cache(saved):
    index(saved)
    n = FakeFaiss.build_count
    second = index(saved)
    assert second.cached and FakeFaiss.build_count == n


def test_force_rebuild(saved):
    index(saved)
    n = FakeFaiss.build_count
    assert not index(saved, force=True).cached
    assert FakeFaiss.build_count == n + 1


def test_bad_index_bytes_rebuilds(saved):
    run = index(saved)
    run.index_path.write_bytes(b"BAD")
    assert not index(saved).cached


def test_bad_index_manifest_rebuilds(saved):
    run = index(saved)
    run.manifest_path.write_text("not json")
    assert not index(saved).cached


def test_missing_index_rebuilds(saved):
    run = index(saved)
    run.index_path.unlink()
    assert not index(saved).cached


def test_mutating_vector_file_detected(saved):
    path = saved / "embeddings" / f"{PAPER}.npy"
    path.write_bytes(b"replaced")
    with pytest.raises(RetrievalError, match="stale or inconsistent"):
        index(saved)


def test_mutating_chunk_source_detected(saved):
    path = saved / "chunks" / f"{PAPER}.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RetrievalError, match="stale or inconsistent"):
        index(saved)


def test_mutating_embedded_metadata_detected(saved):
    path = saved / "embeddings" / f"{PAPER}.manifest.json"
    meta = json.loads(path.read_text())
    meta["chunk_ids"].reverse()
    path.write_text(json.dumps(meta))
    with pytest.raises(RetrievalError, match="stale or inconsistent"):
        index(saved)


def test_invalid_paper_id(saved):
    with pytest.raises(ValueError):
        build_paper_index("../../secrets", data_dir=saved, faiss_backend=FakeFaiss)


def test_missing_embeddings_actionable(tmp_path):
    with pytest.raises(RetrievalError, match="Run chunk-document and embed-document"):
        index(tmp_path)


def test_search_is_grounded_and_ranked(saved):
    result = search(saved)
    assert result.paper_id == PAPER
    assert result.vector_count > 2
    assert len(result.hits) == 5
    assert [hit.rank for hit in result.hits] == [1, 2, 3, 4, 5]
    assert result.hits[0].page_number == 1
    assert "evidence" in result.hits[0].text.lower()
    assert result.hits[0].citation == f"https://arxiv.org/abs/{PAPER}#page=1"
    assert all(result.hits[i].score >= result.hits[i + 1].score for i in range(len(result.hits) - 1))


def test_search_reuses_index(saved):
    index(saved)
    result = search(saved)
    assert result.index_cached


def test_search_autobuilds_index(saved):
    result = search(saved)
    assert not result.index_cached
    assert (saved / "indexes" / f"{PAPER}.faiss").exists()


def test_page_two_retrieval(saved):
    result = search(saved, "cat image", top_k=1)
    assert len(result.hits) == 1
    assert result.hits[0].page_number == 2


def test_search_respects_top_k_and_count(saved):
    result = search(saved, top_k=20)
    assert len(result.hits) == min(20, result.vector_count)


@pytest.mark.parametrize("bad", ["", "   ", "a" * 2001, None])
def test_invalid_query(saved, bad):
    with pytest.raises(RetrievalError, match="query must contain"):
        search(saved, bad)


@pytest.mark.parametrize("bad", [0, -1, 21, True, "5"])
def test_invalid_top_k(saved, bad):
    with pytest.raises(RetrievalError, match="top_k must"):
        search(saved, top_k=bad)


def test_too_long_query_rejected_before_encoding(saved):
    class Short(FakeEncoder):
        max_sequence_length = 3
        def encode_query(self, text):
            raise AssertionError("Query should never be encoded")
    with pytest.raises(RetrievalError, match="token limit"):
        search_paper(PAPER, "agent evidence", data_dir=saved,
                     encoder_factory=Short, faiss_backend=FakeFaiss)


def test_wrong_query_dimension_rejected(saved):
    class Wrong(FakeEncoder):
        dimension = 7
    with pytest.raises(RetrievalError, match="dimension does not match"):
        search_paper(PAPER, "agent", data_dir=saved,
                     encoder_factory=Wrong, faiss_backend=FakeFaiss)


def test_zero_query_vector_rejected(saved):
    class Zero(FakeEncoder):
        def encode_query(self, text):
            return np.zeros(4, dtype=np.float32)
    with pytest.raises(RetrievalError, match="zero norm"):
        search_paper(PAPER, "agent", data_dir=saved,
                     encoder_factory=Zero, faiss_backend=FakeFaiss)


def test_nan_query_vector_rejected(saved):
    class Nan(FakeEncoder):
        def encode_query(self, text):
            return np.array([np.nan, 0, 0, 0])
    with pytest.raises(RetrievalError, match="invalid dimensions or values"):
        search_paper(PAPER, "agent", data_dir=saved,
                     encoder_factory=Nan, faiss_backend=FakeFaiss)


def test_manifest_tampered_index_metric_rebuilds(saved):
    report = index(saved)
    meta = json.loads(report.manifest_path.read_text())
    meta["metric"] = "l2"
    report.manifest_path.write_text(json.dumps(meta))
    assert not index(saved).cached


def test_source_page_offsets_are_from_original_chunks(saved):
    result = search(saved, top_k=3)
    source = json.loads((saved / "chunks" / f"{PAPER}.json").read_text())
    by_id = {chunk["chunk_id"]: chunk for chunk in source["chunks"]}
    for hit in result.hits:
        chunk = by_id[hit.chunk_id]
        assert (hit.start_char, hit.end_char, hit.text, hit.page_number) == (
            chunk["start_char"], chunk["end_char"], chunk["text"], chunk["page_number"]
        )


def test_result_cache_manifest_links_previous_stage(saved):
    report = index(saved)
    embedded = json.loads((saved / "embeddings" / f"{PAPER}.manifest.json").read_text())
    assert report.manifest.vector_file_sha256 == embedded["vector_file_sha256"]
    assert report.manifest.chunk_file_sha256 == embedded["chunk_file_sha256"]


def test_cosine_dot_product_matches_numpy_exactly(saved):
    result = search(saved, top_k=4)
    matrix = np.load(saved / "embeddings" / f"{PAPER}.npy", allow_pickle=False)
    query = FakeEncoder._vector("agent evidence")
    query /= np.linalg.norm(query)
    all_scores = matrix @ query
    expected = np.argsort(-all_scores, kind="stable")[:4]
    chunks = json.loads((saved / "chunks" / f"{PAPER}.json").read_text())["chunks"]
    assert [x.chunk_id for x in result.hits] == [chunks[i]["chunk_id"] for i in expected]


def test_read_only_bundle_no_llm_api_needed(saved):
    assert search(saved, top_k=1).model_name == MODEL_NAME

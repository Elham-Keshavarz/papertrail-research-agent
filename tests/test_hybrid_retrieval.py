"""End-to-end hybrid search on tiny locally-generated source data (no network)."""
import json

import pytest

from papertrail.chunking.pipeline import chunk_saved_document
from papertrail.embedding.pipeline import embed_saved_document
from papertrail.ingestion.pdf_extractor import ExtractedDocument, ExtractedPage
from papertrail.retrieval import RetrievalError
from papertrail.retrieval.hybrid import hybrid_search_paper
from papertrail.retrieval.evaluation import _coverage, compare_benchmark_retrieval, EXPECTED_BENCHMARKS
from test_vector_search import FakeFaiss, FakeEncoder, PAPER


@pytest.fixture
def corpus(tmp_path):
    pages = [
        "An evaluation study compares four question answering benchmarks. " * 5 +
        "HotpotQA and 2WikiMultiHopQA are multi-hop question answering benchmarks. " * 4 +
        "MuSiQue is another complex multi-hop dataset used in evaluation. " * 4,
        "TriviaQA is a factual question answering dataset. " * 8 +
        "The Structured Evidence Assessment agent inspects missing information. " * 5,
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
    chunk_saved_document(PAPER, data_dir=tmp_path, chunk_size=180, overlap=35)
    embed_saved_document(PAPER, data_dir=tmp_path, encoder_factory=FakeEncoder)
    return tmp_path


def run(corpus, query="Which datasets evaluate FAIR-RAG?", **kwargs):
    return hybrid_search_paper(PAPER, query, data_dir=corpus,
                               encoder_factory=FakeEncoder, faiss_backend=FakeFaiss, **kwargs)


def test_hybrid_returns_cited_ranked_hits(corpus):
    r = run(corpus, top_k=5)
    assert r.method == "hybrid_rrf"
    assert len(r.hits) == 5
    assert [h.rank for h in r.hits] == [1, 2, 3, 4, 5]
    assert all(r.hits[i].score >= r.hits[i+1].score for i in range(len(r.hits)-1))
    assert any(h.bm25_rank is not None for h in r.hits)
    assert any(h.dense_rank is not None for h in r.hits)
    assert all(h.citation.endswith(f"#page={h.page_number}") for h in r.hits)
    assert r.dense_candidates <= 20 and r.bm25_candidates <= 60
    assert "not confidence" in r.score_description


def test_rrf_not_raw_score_addition(corpus):
    r = run(corpus, top_k=5)
    for hit in r.hits:
        expected = ((1 / (60 + hit.dense_rank)) if hit.dense_rank else 0) + (
            (1 / (60 + hit.bm25_rank)) if hit.bm25_rank else 0
        )
        assert hit.score == pytest.approx(expected, abs=1e-8)


def test_preserves_exact_chunk_bytes_and_metadata(corpus):
    r = run(corpus, top_k=8)
    doc = json.loads((corpus / "chunks" / f"{PAPER}.json").read_text())
    by_id = {c["chunk_id"]: c for c in doc["chunks"]}
    for hit in r.hits:
        for item in [hit, *hit.context]:
            original = by_id[item.chunk_id]
            assert (item.text, item.page_number, item.start_char, item.end_char) == (
                original["text"], original["page_number"], original["start_char"], original["end_char"]
            )
            assert item.citation == f"https://arxiv.org/abs/{PAPER}#page={item.page_number}"


def test_context_stays_on_same_page_and_not_self(corpus):
    r = run(corpus, top_k=10, context_window=2)
    for hit in r.hits:
        assert len(hit.context) <= 4
        assert all(c.page_number == hit.page_number for c in hit.context)
        assert all(c.chunk_id != hit.chunk_id for c in hit.context)
        assert len({c.chunk_id for c in hit.context}) == len(hit.context)


def test_zero_context_works(corpus):
    assert all(not h.context for h in run(corpus, context_window=0).hits)


def test_one_page_boundary_context(corpus):
    r = run(corpus, top_k=20, context_window=2)
    assert {h.page_number for h in r.hits} == {1, 2}
    assert all(all(c.page_number == h.page_number for c in h.context) for h in r.hits)


def test_caching(corpus):
    first = run(corpus)
    second = run(corpus)
    assert not first.index_cached and second.index_cached


@pytest.mark.parametrize("top_k", [0, 21, True, -1, "2"])
def test_invalid_top_k(corpus, top_k):
    with pytest.raises(RetrievalError, match="top_k"):
        run(corpus, top_k=top_k)


@pytest.mark.parametrize("window", [-1, 3, True, "1"])
def test_invalid_window(corpus, window):
    with pytest.raises(RetrievalError, match="context_window"):
        run(corpus, context_window=window)


@pytest.mark.parametrize("query", ["", " ", "x"*2001, None])
def test_invalid_query(corpus, query):
    with pytest.raises(RetrievalError, match="query must contain"):
        run(corpus, query)


def test_corrupted_chunk_file_blocked(corpus):
    (corpus / "chunks" / f"{PAPER}.json").write_bytes(b"bad")
    with pytest.raises(RetrievalError):
        run(corpus)


def test_evaluation_reports_actual_source_coverage(corpus):
    result = compare_benchmark_retrieval(
        PAPER, data_dir=corpus, top_k=10, encoder_factory=FakeEncoder, faiss_backend=FakeFaiss,
    )
    assert result.expected_names == list(EXPECTED_BENCHMARKS)
    assert result.pdf_extraction_quality_verified is False
    assert result.corpus.covered_count == 4
    assert result.dense.expected_count == result.hybrid.expected_count == 4
    assert result.hybrid.found == [x for x in EXPECTED_BENCHMARKS if x not in result.hybrid.missing]
    assert 0 <= result.hybrid.recall_by_name <= 1
    assert "not semantic relevance" in result.metric_note


def test_coverage_detects_only_real_passage_mentions(corpus):
    r = run(corpus, top_k=20, context_window=2)
    coverage = _coverage(r, EXPECTED_BENCHMARKS)
    combined = "\n".join(p.text for h in r.hits for p in [h, *h.context]).casefold()
    assert coverage.found == [name for name in EXPECTED_BENCHMARKS if name.casefold() in combined]
    assert coverage.covered_count + len(coverage.missing) == 4

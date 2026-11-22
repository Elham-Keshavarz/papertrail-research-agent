"""BM25 relevance, exact technical terms, deterministic ties, and safety."""
import math

import pytest

from papertrail.chunking.chunker import TextChunk
from papertrail.retrieval.bm25 import BM25Index, tokenize


def chunk(index, text):
    return TextChunk(
        chunk_id=f"unit:{index}", paper_id="2510.22344",
        source_url="https://arxiv.org/abs/2510.22344", source_sha256="c" * 64,
        page_number=1, start_char=0, end_char=len(text), text=text,
    )


def test_bm25_finds_rare_dataset_name():
    docs = [chunk(0, "The model was compared with established systems."),
            chunk(1, "MuSiQue 2WikiMultiHopQA benchmarks evaluate the system."),
            chunk(2, "This report discusses evaluation methods.")]
    rows = BM25Index(docs).search("MuSiQue benchmarks")
    assert rows and rows[0].position == 1
    assert rows[0].score > 0
    assert rows[0].rank == 1


def test_case_insensitive_hyphen_tokenizer():
    assert tokenize("FAIR-RAG and 2WikiMultiHopQA on HotpotQA") == ["fair", "rag", "2wikimultihopqa", "hotpotqa"]


def test_bm25_plural_variation():
    index = BM25Index([chunk(0, "The benchmark is difficult."), chunk(1, "Dogs jump")])
    assert index.search("Which benchmarks?", limit=1)[0].position == 0


def test_bm25_unmatched_query_returns_no_false_hits():
    index = BM25Index([chunk(0, "Alpha systems"), chunk(1, "Beta pipelines")])
    assert index.search("quantum") == []
    assert index.search("and the or") == []


def test_bm25_ties_stable():
    index = BM25Index([chunk(0, "TriviaQA"), chunk(1, "TriviaQA"), chunk(2, "Other")])
    assert [h.position for h in index.search("TriviaQA")] == [0, 1]


def test_bm25_long_documents_length_normalization():
    index = BM25Index([chunk(0, "MuSiQue "), chunk(1, "MuSiQue " + "filler " * 50)])
    assert index.search("MuSiQue")[0].position == 0


@pytest.mark.parametrize("k1,b", [(0, 0.75), (-1, 0.75), (1, -0.1), (1, 1.1), (float('nan'), .7)])
def test_invalid_parameters(k1, b):
    with pytest.raises(ValueError):
        BM25Index([chunk(0, "alpha")], k1=k1, b=b)


@pytest.mark.parametrize("limit", [0, True, -5, 3.0, "2"])
def test_invalid_limit(limit):
    with pytest.raises(ValueError):
        BM25Index([chunk(0, "alpha")]).search("alpha", limit=limit)


def test_empty_index_rejected():
    with pytest.raises(ValueError):
        BM25Index([])

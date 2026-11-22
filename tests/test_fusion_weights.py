"""Weighted RRF safety, rank semantics and original 1/1 compatibility."""
from types import SimpleNamespace

import pytest

from papertrail.retrieval import RetrievalError
from papertrail.retrieval.bm25 import BM25Hit
from papertrail.retrieval.fusion import RRF_CONSTANT, fuse_candidates, validate_rrf_weights
from test_hybrid_retrieval import corpus  # noqa: F401
from test_vector_search import FakeEncoder, FakeFaiss, PAPER
from papertrail.retrieval.hybrid import hybrid_search_paper


def dense(cid, rank, score):
    return SimpleNamespace(chunk_id=cid, rank=rank, score=score)


def test_default_rrf_matches_original_equation():
    d = [dense('a', 1, .8), dense('b', 2, .5)]
    s = [BM25Hit(position=1, rank=1, score=8), BM25Hit(position=2, rank=2, score=4)]
    out = fuse_candidates(d, s, {'a': 0, 'b': 1, 'c': 2}, chunk_count=3)
    scores = {x.position: x.score for x in out}
    assert scores[0] == pytest.approx(1 / (RRF_CONSTANT + 1))
    assert scores[1] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    assert out[0].position == 1


def test_two_weights_can_reorder_candidate_positions_without_changing_input():
    d = [dense('a', 1, .2), dense('b', 2, .1)]
    s = [BM25Hit(position=1, rank=1, score=3), BM25Hit(position=0, rank=2, score=2)]
    position = {'a': 0, 'b': 1}
    dense_first = fuse_candidates(d, s, position, chunk_count=2, dense_weight=4, bm25_weight=1)
    sparse_first = fuse_candidates(d, s, position, chunk_count=2, dense_weight=1, bm25_weight=4)
    assert dense_first[0].position == 0
    assert sparse_first[0].position == 1
    assert d[0].rank == 1 and s[0].rank == 1


def test_zero_weight_suppresses_exclusive_candidates():
    d = [dense('a', 1, .2)]
    s = [BM25Hit(position=1, rank=1, score=8)]
    positions = {'a': 0, 'b': 1}
    assert [h.position for h in fuse_candidates(d, s, positions, chunk_count=2, dense_weight=0)] == [1]
    assert [h.position for h in fuse_candidates(d, s, positions, chunk_count=2, bm25_weight=0)] == [0]


@pytest.mark.parametrize('a,b', [(0, 0), (-1, 1), (1, -1), (11, 1), (float('nan'), 1),
                                  (float('inf'), 1), (True, 1), ('1', 1), (None, 1)])
def test_invalid_weights_fail_closed(a, b):
    with pytest.raises(RetrievalError, match='weight'):
        validate_rrf_weights(a, b)


@pytest.mark.parametrize('bad_dense,bad_sparse,pattern', [
    ([dense('unknown', 1, .3)], [], 'Dense candidates'),
    ([dense('a', 1, .3), dense('a', 2, .2)], [], 'Dense candidates'),
    ([], [BM25Hit(position=4, rank=1, score=1)], 'BM25 candidate'),
    ([], [BM25Hit(position=0, rank=1, score=1), BM25Hit(position=0, rank=2, score=2)], 'BM25 candidate'),
])
def test_fusion_rejects_invalid_candidates(bad_dense, bad_sparse, pattern):
    with pytest.raises(RetrievalError, match=pattern):
        fuse_candidates(bad_dense, bad_sparse, {'a': 0}, chunk_count=1)


def test_hybrid_weights_are_reported_and_citations_preserved(corpus):
    result = hybrid_search_paper(PAPER, "Which datasets were evaluated?", data_dir=corpus,
        top_k=5, dense_weight=2, bm25_weight=1, encoder_factory=FakeEncoder, faiss_backend=FakeFaiss)
    assert (result.dense_weight, result.bm25_weight) == (2, 1)
    assert all(hit.citation.endswith(f'#page={hit.page_number}') for hit in result.hits)
    assert all(hit.score > 0 for hit in result.hits)


def test_hybrid_rejects_zero_zero_before_loading_cache(corpus):
    with pytest.raises(RetrievalError, match='weight'):
        hybrid_search_paper(PAPER, 'question', data_dir=corpus, dense_weight=0, bm25_weight=0)

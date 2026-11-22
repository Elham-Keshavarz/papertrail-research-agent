"""Page-capping is transparent and never edits evidence, source rank or score."""
from dataclasses import replace

import pytest

from papertrail.retrieval import RetrievalError
from papertrail.retrieval.fusion import FusedCandidate, select_page_diverse, validate_page_cap
from papertrail.retrieval.hybrid import hybrid_search_paper
from test_hybrid_retrieval import corpus  # noqa: F401
from test_vector_search import FakeFaiss, FakeEncoder, PAPER


def item(i):
    return FusedCandidate(position=i, score=1/(60+i+1), dense_rank=i+1,
                          dense_score=.9-i*.1, bm25_rank=None, bm25_score=None)


def test_page_cap_is_stable_and_no_modifications():
    items = [item(i) for i in range(6)]
    pages = [1,1,1,2,2,3]
    selected = select_page_diverse(items, pages, top_k=4, max_hits_per_page=2)
    assert [x.position for x in selected] == [0,1,3,4]
    assert selected == [items[i] for i in (0,1,3,4)]
    assert select_page_diverse(items, pages, top_k=4) == items[:4]
    assert [x.position for x in select_page_diverse(items, pages, top_k=5, max_hits_per_page=1)] == [0,3,5]


@pytest.mark.parametrize('invalid', [True, -1, 6, 1.5, '2'])
def test_page_cap_rejects_invalid(invalid):
    with pytest.raises(RetrievalError, match='max_hits_per_page'):
        validate_page_cap(invalid)


def test_page_cap_rejects_corrupt_position():
    with pytest.raises(RetrievalError, match='outside'):
        select_page_diverse([item(3)], [1,2], top_k=1)


def test_e2e_capped_hybrid_fewer_repeats_on_synthetic_corpus(corpus):
    result = hybrid_search_paper(PAPER,'Which datasets were evaluated?',data_dir=corpus,
        top_k=10,max_hits_per_page=2,encoder_factory=FakeEncoder,faiss_backend=FakeFaiss)
    assert result.max_hits_per_page == 2
    assert len([h for h in result.hits if h.page_number == 1]) <= 2
    assert len([h for h in result.hits if h.page_number == 2]) <= 2
    assert len(result.hits) <= 4
    assert all(h.citation.endswith(str(h.page_number)) for h in result.hits)

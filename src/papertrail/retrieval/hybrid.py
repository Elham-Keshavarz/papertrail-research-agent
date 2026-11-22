"""Hybrid FAISS + BM25 retrieval, reciprocal rank fusion, and cited context.

RRF ranks the *union* of candidate passages; it never fuses raw cosine and
BM25 scores. Neighbor expansion preserves each chunk's original citation and
never crosses a page boundary. Local retrieval only: no LLM or external API.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from papertrail.embedding.encoder import Encoder
from papertrail.embedding.pipeline import load_saved_embedding_bundle
from papertrail.ingestion.pdf_downloader import validate_paper_id
from .bm25 import BM25Index
from .fusion import fuse_candidates, validate_rrf_weights, select_page_diverse, validate_page_cap
from .faiss_store import RetrievalError
from .search import MAX_QUERY_CHARACTERS, MAX_TOP_K, search_paper

DEFAULT_DENSE_CANDIDATES = 20  # Preserves the existing FAISS search API limit.
DEFAULT_SPARSE_CANDIDATES = 60


class EvidencePassage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    page_number: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str
    citation: str


class HybridHit(EvidencePassage):
    rank: int = Field(ge=1)
    score: float = Field(gt=0)
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    context: list[EvidencePassage] = Field(default_factory=list)


class HybridResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    query: str
    method: str = "hybrid_rrf"
    score_description: str = "RRF ranking score; not confidence and not cosine similarity"
    model_name: str
    model_revision: str
    vector_count: int = Field(gt=0)
    top_k: int = Field(ge=1)
    context_window: int = Field(ge=0, le=2)
    dense_candidates: int = Field(ge=0)
    bm25_candidates: int = Field(ge=0)
    index_cached: bool
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    max_hits_per_page: int = 0
    hits: list[HybridHit]


def _passage(chunk: Any) -> EvidencePassage:
    return EvidencePassage(
        chunk_id=chunk.chunk_id, page_number=chunk.page_number,
        start_char=chunk.start_char, end_char=chunk.end_char,
        text=chunk.text, citation=f"{chunk.source_url}#page={chunk.page_number}",
    )


def _context(chunks: list[Any], position: int, window: int) -> list[EvidencePassage]:
    """Return ordered immediate neighbors on the *same page*, excluding seed."""
    center = chunks[position]
    indices = [i for i in range(max(0, position-window), min(len(chunks), position+window+1))
               if i != position and chunks[i].page_number == center.page_number]
    return [_passage(chunks[i]) for i in indices]


def _validate_args(query: str, top_k: int, context_window: int) -> None:
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARACTERS:
        raise RetrievalError(f"query must contain 1 to {MAX_QUERY_CHARACTERS} characters")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise RetrievalError(f"top_k must be an integer from 1 to {MAX_TOP_K}")
    if isinstance(context_window, bool) or not isinstance(context_window, int) or not 0 <= context_window <= 2:
        raise RetrievalError("context_window must be 0, 1 or 2")


def hybrid_search_paper(
    paper_id: str, query: str, *, data_dir: Path, top_k: int = 5,
    context_window: int = 1, dense_weight: float = 1.0, bm25_weight: float = 1.0,
    max_hits_per_page: int = 0,
    encoder_factory: Callable[[str, str], Encoder] | None = None,
    faiss_backend: Any | None = None,
) -> HybridResult:
    paper_id = validate_paper_id(paper_id)
    _validate_args(query, top_k, context_window)
    dense_weight, bm25_weight = validate_rrf_weights(dense_weight, bm25_weight)
    validate_page_cap(max_hits_per_page)
    query = query.strip()
    try:
        bundle = load_saved_embedding_bundle(paper_id, data_dir=data_dir)
    except (ValueError, OSError) as exc:
        raise RetrievalError(str(exc)) from exc
    chunks = bundle.document.chunks
    if len(chunks) != bundle.manifest.vector_count:
        raise RetrievalError("Chunk-to-vector alignment mismatch")
    # Reuse existing vetted query encoder, index integrity and cosine validation.
    dense = search_paper(
        paper_id, query, data_dir=data_dir,
        top_k=min(DEFAULT_DENSE_CANDIDATES, len(chunks)),
        encoder_factory=encoder_factory, faiss_backend=faiss_backend,
    )
    positions_by_id = {chunk.chunk_id: position for position, chunk in enumerate(chunks)}
    if len(positions_by_id) != len(chunks):
        raise RetrievalError("Duplicate chunk identifiers are not allowed")
    bm25 = BM25Index(chunks).search(query, limit=min(DEFAULT_SPARSE_CANDIDATES, len(chunks)))

    ranked = select_page_diverse(
        fuse_candidates(
            dense.hits, bm25, positions_by_id, chunk_count=len(chunks),
            dense_weight=dense_weight, bm25_weight=bm25_weight,
        ), [c.page_number for c in chunks],
        top_k=top_k, max_hits_per_page=max_hits_per_page,
    )
    hits: list[HybridHit] = []
    for rank, candidate in enumerate(ranked, 1):
        chunk = chunks[candidate.position]
        hits.append(HybridHit(
            **_passage(chunk).model_dump(),
            rank=rank, score=round(candidate.score, 8),
            dense_rank=candidate.dense_rank, dense_score=candidate.dense_score,
            bm25_rank=candidate.bm25_rank, bm25_score=candidate.bm25_score,
            context=_context(chunks, candidate.position, context_window),
        ))
    return HybridResult(
        paper_id=paper_id, query=query, model_name=bundle.manifest.model_name,
        model_revision=bundle.manifest.model_revision,
        vector_count=len(chunks), top_k=top_k, context_window=context_window,
        dense_candidates=len(dense.hits), bm25_candidates=len(bm25),
        index_cached=dense.index_cached, dense_weight=dense_weight, bm25_weight=bm25_weight,
        max_hits_per_page=max_hits_per_page, hits=hits,
    )

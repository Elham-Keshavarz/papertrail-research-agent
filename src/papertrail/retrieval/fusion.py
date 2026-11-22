"""Pure weighted reciprocal-rank fusion; reusable for search and offline tuning.

The only inputs are independently retrieved, already ranked candidate lists. No
relevance judgments enter fusion, and BM25 scores are never added to cosines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from .bm25 import BM25Hit
from .faiss_store import RetrievalError

RRF_CONSTANT = 60
MAX_RRF_WEIGHT = 10.0


@dataclass(frozen=True)
class FusedCandidate:
    position: int
    score: float
    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None


def validate_rrf_weights(dense_weight: float, bm25_weight: float) -> tuple[float, float]:
    values = (dense_weight, bm25_weight)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or not 0 <= x <= MAX_RRF_WEIGHT for x in values):
        raise RetrievalError(f"RRF weights must be finite numbers in 0..{MAX_RRF_WEIGHT:g}")
    if not any(values):
        raise RetrievalError("At least one RRF weight must be positive")
    return float(dense_weight), float(bm25_weight)


def fuse_candidates(
    dense_hits: Sequence[Any], sparse_hits: Sequence[BM25Hit],
    positions_by_id: Mapping[str, int], *, chunk_count: int,
    dense_weight: float = 1.0, bm25_weight: float = 1.0,
) -> list[FusedCandidate]:
    """Return deterministically ranked candidates, without reading ground truth.

    A zero weight suppresses that retriever's exclusive candidates. Its ranks
    and raw scores never contribute to the fused score. A `position` refers to
    the original immutable on-disk chunk sequence.
    """
    dw, bw = validate_rrf_weights(dense_weight, bm25_weight)
    rows: dict[int, dict[str, float | int | None]] = {}
    seen_dense: set[int] = set()
    seen_sparse: set[int] = set()

    for hit in dense_hits:
        pos = positions_by_id.get(hit.chunk_id)
        if pos is None or not 0 <= pos < chunk_count or pos in seen_dense:
            raise RetrievalError("Dense candidates are missing, duplicated or misaligned")
        if isinstance(hit.rank, bool) or not isinstance(hit.rank, int) or hit.rank < 1:
            raise RetrievalError("Dense candidate rank is invalid")
        seen_dense.add(pos)
        if dw:
            rows[pos] = {
                "score": dw / (RRF_CONSTANT + hit.rank),
                "dense_rank": hit.rank, "dense_score": float(hit.score),
                "bm25_rank": None, "bm25_score": None,
            }

    for hit in sparse_hits:
        pos = hit.position
        if isinstance(pos, bool) or not isinstance(pos, int) or not 0 <= pos < chunk_count or pos in seen_sparse:
            raise RetrievalError("BM25 candidate position is invalid or duplicated")
        if isinstance(hit.rank, bool) or not isinstance(hit.rank, int) or hit.rank < 1:
            raise RetrievalError("BM25 candidate rank is invalid")
        seen_sparse.add(pos)
        if bw:
            if pos not in rows:
                rows[pos] = {
                    "score": 0.0, "dense_rank": None, "dense_score": None,
                    "bm25_rank": None, "bm25_score": None,
                }
            rows[pos]["score"] += bw / (RRF_CONSTANT + hit.rank)
            rows[pos]["bm25_rank"] = hit.rank
            rows[pos]["bm25_score"] = float(hit.score)

    ordered = sorted(rows.items(), key=lambda pair: (
        -float(pair[1]["score"]),
        -float(pair[1]["bm25_score"] or 0.0),
        -float(pair[1]["dense_score"] if pair[1]["dense_score"] is not None else -1.0),
        pair[0],
    ))
    return [FusedCandidate(position=pos, score=float(item["score"]),
                           dense_rank=item["dense_rank"], dense_score=item["dense_score"],
                           bm25_rank=item["bm25_rank"], bm25_score=item["bm25_score"])
            for pos, item in ordered]


def validate_page_cap(max_hits_per_page: int) -> int:
    """0 keeps every candidate; 1..5 caps page repetition among primary hits."""
    if (isinstance(max_hits_per_page, bool) or not isinstance(max_hits_per_page, int)
            or not 0 <= max_hits_per_page <= 5):
        raise RetrievalError("max_hits_per_page must be an integer from 0 to 5")
    return max_hits_per_page


def select_page_diverse(
    ranked: Sequence[FusedCandidate], page_by_position: Sequence[int], *,
    top_k: int, max_hits_per_page: int = 0,
) -> list[FusedCandidate]:
    """Optional stable filtering, never changes underlying RRF scores or citations."""
    cap = validate_page_cap(max_hits_per_page)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise RetrievalError("top_k must be positive")
    counts: dict[int, int] = {}
    selected = []
    for candidate in ranked:
        if not 0 <= candidate.position < len(page_by_position):
            raise RetrievalError("Candidate position outside verified corpus")
        page = page_by_position[candidate.position]
        if not isinstance(page, int) or page < 1:
            raise RetrievalError("Invalid candidate page metadata")
        if cap and counts.get(page, 0) >= cap:
            continue
        selected.append(candidate)
        counts[page] = counts.get(page, 0) + 1
        if len(selected) >= top_k:
            break
    return selected

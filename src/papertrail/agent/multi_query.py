"""Phase 4.2.1 deterministic, bounded cross-query fusion for preview-only research.

The input lists are individually validated primary hits from the existing retrieval
adapter. Cross-query RRF scores are *ranking heuristics*, not relevance/confidence.
This module never selects a winning policy from benchmark labels.
"""
from __future__ import annotations

import math

_QUERY_STYLES = ("baseline", "focused")
_RRF_K = 60


def fuse_validated_queries(branches: dict[str, list[dict]], *, limit: int) -> list[dict]:
    """RRF across two ranked lists; exact chunk-ID dedup, fixed output budget.

    Ties resolve by the best source rank, then baseline-first and chunk ID,
    independent of external labels. Contradictory source payloads fail closed.
    """
    if type(limit) is not int or not 1 <= limit <= 10 or set(branches) != set(_QUERY_STYLES):
        raise ValueError("Invalid multi-query budget or branch set")
    contents: dict[str, dict] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name in _QUERY_STYLES:
        hits = branches[name]
        if not isinstance(hits, list) or len(hits) > limit:
            raise ValueError("Unbounded multi-query source result")
        for hit in hits:
            chunk_id = hit["chunk_id"]
            if chunk_id in contents:
                previous = contents[chunk_id]
                if any(previous[key] != hit[key] for key in ("page_number", "citation", "text", "context")):
                    raise ValueError("Conflicting source provenance across queries")
            else:
                contents[chunk_id] = hit
            ranks.setdefault(chunk_id, {})[name] = hit["rank"]

    def ordering(chunk_id: str) -> tuple:
        branch_ranks = ranks[chunk_id]
        score = sum(1 / (_RRF_K + r) for r in branch_ranks.values())
        # Stable tie resolution; numerical RRF is not a semantic score.
        return (-score, min(branch_ranks.values()),
                0 if "baseline" in branch_ranks else 1, chunk_id)

    result = []
    for idx, chunk_id in enumerate(sorted(contents, key=ordering)[:limit], start=1):
        hit = contents[chunk_id]
        score = sum(1 / (_RRF_K + r) for r in ranks[chunk_id].values())
        if not math.isfinite(score):
            raise ValueError("Invalid cross-query fusion score")
        result.append({**hit, "rank": idx, "rrf_score": round(score, 10)})
    return result


def fuse_budgeted_queries(branches: dict[str, list[dict]], *, limit: int) -> list[dict]:
    """Budget-aware, provenance-checked focused-first interleaving.

    A fixed per-facet output cap is enforced. Interleaving provides both search
    branches an opportunity to contribute *distinct* chunks before additional
    chunks from either branch consume the budget. Focused starts each round
    because it is the explicitly opted-in query diversification path, not
    because any benchmark label or ground-truth chunk was consulted.

    This is a rank-list coverage heuristic, NOT semantic reranking or relevance
    validation. It may underperform RRF and is never used for live generation.
    """
    # The existing fusion contract performs branch count/budget and
    # provenance-consistency validation before any evidence is admitted.
    fuse_validated_queries(branches, limit=limit)
    ranks: dict[str, dict[str, int]] = {}
    contents: dict[str, dict] = {}
    for style in _QUERY_STYLES:
        for hit in branches[style]:
            cid = hit["chunk_id"]
            contents[cid] = hit
            ranks.setdefault(cid, {})[style] = hit["rank"]

    selected: list[str] = []
    seen: set[str] = set()
    for rank_position in range(limit):
        for style in ("focused", "baseline"):
            if rank_position < len(branches[style]):
                cid = branches[style][rank_position]["chunk_id"]
                if cid not in seen:
                    seen.add(cid)
                    selected.append(cid)
                    if len(selected) == limit:
                        break
        if len(selected) == limit:
            break
    # Duplicate-heavy lists can exhaust without filling, which is legitimate.
    result = []
    for index, cid in enumerate(selected, 1):
        score = sum(1 / (_RRF_K + rank) for rank in ranks[cid].values())
        if not math.isfinite(score):
            raise ValueError("Invalid budget-aware fusion score")
        result.append({**contents[cid], "rank": index, "rrf_score": round(score, 10)})
    return result

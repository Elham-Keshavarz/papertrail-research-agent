"""Conservative Phase 3.4 evidence diagnostics, NOT factual verification.

All decisions are made from bounded, source-attributed retrieval metadata.
Lexical/structural coverage is not interpreted as entailment or answer quality.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


def assess_evidence(plan: list[dict], evidence: list[dict], *, min_hits_per_step: int = 2,
                    top_k: int = 3, source_audit: dict | None = None,
                    retry_attempts: dict[str, int] | None = None) -> dict[str, Any]:
    """Summarize independent planned subquestions, pages, and evidence diversity.

    Each subquestion is assessed separately: retrieved passages cannot compensate
    for a missing facet. No score here establishes semantic relevance or truth.
    """
    if not 1 <= min_hits_per_step <= 10 or not 1 <= top_k <= 10:
        raise ValueError("Assessment thresholds must be between 1 and 10")
    steps = [step for step in plan if step.get("proposed_tool") == "hybrid_retrieval"]
    found = {entry["step_id"]: entry.get("hits", []) for entry in evidence}
    threshold = min(min_hits_per_step, top_k)
    all_chunks: list[str] = []
    all_pages: set[int] = set()
    diagnostics: list[dict] = []
    gaps: list[str] = []
    warnings: list[str] = []
    attempts = retry_attempts or {}

    for step in steps:
        step_id = step["id"]
        hits = found.get(step_id, [])
        unique_chunks = {hit["chunk_id"] for hit in hits}
        pages = sorted({hit["page_number"] for hit in hits})
        all_pages.update(pages)
        all_chunks.extend(hit["chunk_id"] for hit in hits)
        enough = len(unique_chunks) >= threshold
        if not enough:
            gaps.append(step_id)
        if len(pages) == 1 and len(unique_chunks) > 1:
            warnings.append(f"{step_id}: multiple chunks originate from one PDF page")
        if len(unique_chunks) < len(hits):
            warnings.append(f"{step_id}: duplicate primary chunk IDs were returned")
        diagnostics.append({
            "step_id": step_id,
            "query": step["search_query"],
            "primary_hit_count": len(hits),
            "unique_primary_chunks": len(unique_chunks),
            "unique_pages": pages,
            "meets_structural_threshold": enough,
            "retry_attempts": attempts.get(step_id, 0),
            "semantic_relevance_verified": False,
        })

    counts = Counter(all_chunks)
    repeated = sorted(chunk_id for chunk_id, n in counts.items() if n > 1)
    if repeated:
        warnings.append("Some source chunks are reused across different subquestions")
    audit = source_audit or {}
    if not audit.get("pdf_extraction_quality_verified", False):
        warnings.append("Source PDF extraction/read-order accuracy is not independently verified")
    if not steps:
        decision = "no_retrieval_steps"
    elif gaps:
        decision = "needs_more_retrieval"
    else:
        decision = "structurally_covered_unverified"
    return {
        "decision": decision,
        "method": "deterministic_structural_proxy_v1",
        "minimum_unique_chunks_per_step": threshold,
        "planned_subquestions": len(steps),
        "subquestions_meeting_threshold": len(steps) - len(gaps),
        "gap_step_ids": gaps,
        "per_subquestion": diagnostics,
        "total_primary_hits": sum(len(entry.get("hits", [])) for entry in evidence),
        "distinct_primary_chunks": len(set(all_chunks)),
        "cross_subquestion_duplicate_chunk_ids": repeated[:30],
        "distinct_source_pages": sorted(all_pages),
        "citation_format_validated_by_retrieval": any(entry.get("hits") for entry in evidence),
        "pdf_extraction_quality_verified": bool(audit.get("pdf_extraction_quality_verified", False)),
        "warnings": warnings,
        "factually_sufficient": None,
        "semantic_coverage_verified": False,
        "contradiction_analysis_performed": False,
        "answer_generation_allowed": False,
        "note": (
            "Structural retrieval diagnostics only. A threshold hit is not proof of relevance, "
            "entailment, factual sufficiency, absence of contradictions, or answer correctness. "
            "Source page/citation and extraction fidelity require independent review."
        ),
    }


def refinement_query(*, original_question: str, planned_query: str) -> str | None:
    """One bounded query simplification using *only* the user's own question.

    No model-authored facts or untrusted PDF instructions enter the retry query.
    The fallback is useful for overly verbose plan-generated queries but does not
    guarantee more relevant passages.
    """
    original = " ".join(original_question.split()).strip()
    planned = " ".join(planned_query.split()).strip()
    if not original or original.casefold() == planned.casefold():
        return None
    return original[:500]

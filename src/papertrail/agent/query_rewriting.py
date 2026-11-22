"""Deterministic, opt-in, query-only facet expansion (no LLM or inferred facts).

This is an exploratory retrieval intervention, NOT a source-grounded claim or
an automatically selected production policy. The lexicon names possible search
terms; its presence does not assert that the paper contains those terms.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

# General retrieval keywords; no paper-specific labels, page numbers or answers.
_FACET_HINTS: dict[str, str] = {
    "comparison": "baseline standard approach differences retrieval reasoning comparison",
    "datasets": "evaluation datasets benchmark corpus experiments",
    "sample_size": "evaluation sample size examples instances experimental setup",
    "metrics": "evaluation metrics exact match F1 accuracy precision recall performance",
    "limitations": "component-level failures architectural failures limitations bottlenecks retrieval failure",
    "methodology": "method architecture pipeline components implementation ablation",
    "iterations": "iterative retrieval refinement cycles maximum iterations",
    "evidence": "evidence sufficiency source support citation verification",
    "definition": "definition overview formulation explanation",
    "general": "research findings method evaluation conclusions",
}
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "compare", "compared",
    "describe", "discuss", "do", "does", "explain", "for", "from", "how", "in", "into", "is",
    "its", "of", "on", "or", "paper", "report", "reported", "the", "their", "this",
    "to", "use", "used", "versus", "what", "which", "with", "were", "why",
    "limitations", "limitation", "metrics", "metric", "evaluation", "evaluated",
    "datasets", "dataset", "methods", "method", "approaches", "approach", "performance",
})


def _topic(question: str) -> str:
    """Use only tokens already in the question; avoid guessed entities and facts."""
    # Named technical methods stand out as uppercase / hyphenated acronyms.
    names = re.findall(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b|\b[A-Z]{2,}(?:[0-9]+)?\b", question)
    names = [word for word in names if word.lower() not in _STOP and len(word) <= 50]
    # FAIR-RAG subsumes the generic 'RAG' in this example. The rule is generic
    # for suffix-method names, not keyed to a particular paper.
    kept: list[str] = []
    for word in names:
        if any(word.casefold() == already.casefold() or
               (already.casefold().endswith("-" + word.casefold())) for already in kept):
            continue
        kept.append(word)
    if kept:
        return " ".join(kept[:3])
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", question)
    useful: list[str] = []
    for word in words:
        if word.casefold() not in _STOP and word.casefold() not in {w.casefold() for w in useful}:
            useful.append(word)
    return " ".join(useful[:7]) or " ".join(question.split()[:9])


def focused_queries(question: str, intents: Sequence[str]) -> list[str]:
    """Return one bounded search query per facet, independent of source contents."""
    if not isinstance(question, str) or not 12 <= len(question.strip()) <= 1000:
        raise ValueError("Invalid original question")
    if not intents or len(intents) > 5 or any(facet not in _FACET_HINTS for facet in intents):
        raise ValueError("Unknown or unbounded facet list")
    subject = _topic(question)
    results = []
    for facet in intents:
        candidate = f"{subject} {_FACET_HINTS[facet]}"
        # Search terms are data. Never execute or interpret question contents.
        candidate = " ".join(candidate.split())[:280].strip()
        results.append(candidate)
    return results

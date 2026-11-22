"""Transparent lexical coverage diagnostic on a fixed, user-verified question.

Coverage means an expected dataset name occurs in an actual cited retrieved
passage; it is NOT a measure of answer correctness, scientific validity or
claim support. It must never be treated as a benchmark of unseen QA accuracy.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from papertrail.embedding.encoder import Encoder
from papertrail.embedding.pipeline import load_saved_embedding_bundle
from .faiss_store import RetrievalError
from .hybrid import HybridResult, hybrid_search_paper
from .search import SearchResult, search_paper

BENCHMARK_QUERY = "Which datasets were used to evaluate FAIR-RAG in the experiments?"
EXPECTED_BENCHMARKS = ("HotpotQA", "2WikiMultiHopQA", "MuSiQue", "TriviaQA")


class Coverage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    found: list[str]
    missing: list[str]
    covered_count: int = Field(ge=0)
    expected_count: int = Field(gt=0)
    recall_by_name: float = Field(ge=0, le=1)
    cited_pages: list[int]


class RetrievalComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    query: str
    top_k: int
    expected_names: list[str]
    corpus: Coverage
    dense: Coverage
    hybrid: Coverage
    metric_note: str = "Name coverage in cited excerpts, not semantic relevance or answer correctness"
    pdf_extraction_quality_verified: bool = False


def _collect_passages(result: SearchResult | HybridResult) -> list[Any]:
    passages: list[Any] = []
    for hit in result.hits:
        passages.append(hit)
        if isinstance(result, HybridResult):
            passages.extend(hit.context)
    return passages


def _coverage_passages(passages: list[Any], names: tuple[str, ...]) -> Coverage:
    content = "\n".join(p.text for p in passages).casefold()
    found = [name for name in names if name.casefold() in content]
    pages = sorted({p.page_number for p in passages if any(name.casefold() in p.text.casefold() for name in found)})
    return Coverage(
        found=found, missing=[name for name in names if name not in found],
        covered_count=len(found), expected_count=len(names),
        recall_by_name=round(len(found) / len(names), 4), cited_pages=pages,
    )


def _coverage(result: SearchResult | HybridResult, names: tuple[str, ...]) -> Coverage:
    return _coverage_passages(_collect_passages(result), names)


def compare_benchmark_retrieval(
    paper_id: str, *, data_dir: Path, top_k: int = 10, context_window: int = 1,
    encoder_factory: Callable[[str, str], Encoder] | None = None,
    faiss_backend: Any | None = None,
) -> RetrievalComparison:
    # No hard-coded expected answer is injected into either retrieval query.
    dense = search_paper(
        paper_id, BENCHMARK_QUERY, data_dir=data_dir, top_k=top_k,
        encoder_factory=encoder_factory, faiss_backend=faiss_backend,
    )
    hybrid = hybrid_search_paper(
        paper_id, BENCHMARK_QUERY, data_dir=data_dir, top_k=top_k,
        context_window=context_window,
        encoder_factory=encoder_factory, faiss_backend=faiss_backend,
    )
    try:
        bundle = load_saved_embedding_bundle(paper_id, data_dir=data_dir)
    except (ValueError, OSError) as exc:
        raise RetrievalError(str(exc)) from exc
    return RetrievalComparison(
        paper_id=paper_id, query=BENCHMARK_QUERY, top_k=top_k,
        expected_names=list(EXPECTED_BENCHMARKS),
        corpus=_coverage_passages(bundle.document.chunks, EXPECTED_BENCHMARKS),
        dense=_coverage(dense, EXPECTED_BENCHMARKS),
        hybrid=_coverage(hybrid, EXPECTED_BENCHMARKS),
    )

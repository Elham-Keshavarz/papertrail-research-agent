"""Offline, reproducible *page-proxy* ranking evaluation for saved paper chunks.

The relevance judgments refer to PDF pages, not individually annotated chunks.
Therefore a hit on a labeled page is only a *proxy* for passage relevance, and
these metrics must not be presented as answer correctness or evidence fidelity.
Only primary retrieved chunks are scored; hybrid's expanded context is excluded
from scoring to retain identical evaluation units across methods.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from papertrail.embedding.encoder import Encoder, SentenceTransformerEncoder
from papertrail.embedding.pipeline import load_saved_embedding_bundle
from papertrail.ingestion.pdf_downloader import validate_paper_id
from .bm25 import BM25Index
from .faiss_store import RetrievalError
from .hybrid import hybrid_search_paper
from .fusion import validate_rrf_weights, validate_page_cap
from .search import search_paper

MAX_SUITE_BYTES = 100_000
METHODS = ("bm25", "faiss", "hybrid")
DEFAULT_K_VALUES = (1, 3, 5, 10)
MAX_EVAL_QUERIES = 40
MAX_K = 20


class RetrievalLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_number: int = Field(ge=1)
    note: str = Field(min_length=8, max_length=500)


class RetrievalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,55}$")
    question: str = Field(min_length=12, max_length=400)
    labels: list[RetrievalLabel] = Field(min_length=1, max_length=15)

    @model_validator(mode="after")
    def labels_unique(self) -> "RetrievalCase":
        numbers = [label.page_number for label in self.labels]
        if len(numbers) != len(set(numbers)):
            raise ValueError("Duplicate relevant pages in a question")
        return self


class RetrievalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = Field(ge=1, le=1)
    suite_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,55}$")
    paper_id: str
    annotation_status: str = Field(pattern=r"^(provisional|reviewed)$")
    annotation_unit: str = Field(pattern=r"^page$")
    provenance: str = Field(min_length=12, max_length=700)
    cases: list[RetrievalCase] = Field(min_length=1, max_length=MAX_EVAL_QUERIES)

    @model_validator(mode="after")
    def validate_cases(self) -> "RetrievalSuite":
        validate_paper_id(self.paper_id)
        ids = [item.id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate query identifiers")
        return self


class RankedPage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rank: int = Field(ge=1)
    page_number: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)


class RankingMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    mrr_at_k: float = Field(ge=0, le=1)
    hit_at_k: float = Field(ge=0, le=1)
    relevant_hits: int = Field(ge=0)
    distinct_relevant_pages: int = Field(ge=0)
    returned_hits: int = Field(ge=0)


class QueryEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    question: str
    relevant_pages: list[int]
    top_hits: dict[str, list[RankedPage]]
    metrics: dict[str, dict[str, RankingMetrics]]


class MetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    mrr_at_k: float = Field(ge=0, le=1)
    hit_at_k: float = Field(ge=0, le=1)


class RetrievalEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suite_id: str
    paper_id: str
    annotation_status: str
    annotation_unit: str = "page"
    corpus_chunk_sha256: str
    vector_count: int = Field(gt=0)
    embedding_model: str
    model_revision: str
    query_count: int = Field(gt=0)
    k_values: list[int]
    methods: list[str]
    macro_results: dict[str, dict[str, MetricSummary]]
    query_results: list[QueryEvaluation]
    methodology_note: str = (
        "Page-proxy relevance: all primary hits on a labeled PDF page count as relevant chunks. "
        "P@K uses K as denominator; Recall@K deduplicates labeled pages, MRR@K scores the first "
        "labeled-page hit. Hybrid neighbor context is NOT scored. Qrels may be incomplete, so "
        "these are preliminary ranking diagnostics, NOT evidence correctness or QA accuracy."
    )
    pdf_extraction_quality_verified: bool = False
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    max_hits_per_page: int = 0


def load_retrieval_suite(path: Path) -> RetrievalSuite:
    path = Path(path)
    try:
        if not path.is_file():
            raise RetrievalError(f"Evaluation suite not found: {path}")
        if path.stat().st_size > MAX_SUITE_BYTES:
            raise RetrievalError("Evaluation suite exceeds size limit")
        return RetrievalSuite.model_validate_json(path.read_bytes())
    except RetrievalError:
        raise
    except (OSError, ValueError, ValidationError) as exc:
        raise RetrievalError(f"Invalid evaluation suite: {exc}") from exc


def parse_k_values(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = value.split(",")
        if not parts or any(not part.strip().isdigit() for part in parts):
            raise RetrievalError("k-values must be comma-separated integers, e.g. 1,3,5,10")
        ks = [int(x.strip()) for x in parts]
    else:
        ks = list(value)
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K for k in ks):
        raise RetrievalError(f"k-values must be integers in 1..{MAX_K}")
    if sorted(set(ks)) != ks:
        raise RetrievalError("k-values must be unique and strictly increasing")
    return tuple(ks)


def score_at_k(hits: Sequence[RankedPage], relevant_pages: set[int], k: int) -> RankingMetrics:
    """Score primary *chunks* with proxy page labels. Missing slots are irrelevant."""
    if not relevant_pages or isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("A positive k and at least one relevant page are required")
    truncated = list(hits[:k])
    matched = [hit for hit in truncated if hit.page_number in relevant_pages]
    distinct = {hit.page_number for hit in matched}
    first_rank = next((i for i, hit in enumerate(truncated, 1) if hit.page_number in relevant_pages), None)
    return RankingMetrics(
        precision_at_k=round(len(matched) / k, 6),
        recall_at_k=round(len(distinct) / len(relevant_pages), 6),
        mrr_at_k=round(1 / first_rank, 6) if first_rank is not None else 0.0,
        hit_at_k=float(first_rank is not None),
        relevant_hits=len(matched),
        distinct_relevant_pages=len(distinct),
        returned_hits=len(truncated),
    )


def _rank_pages(rows: Sequence[Any], *, expected_ids: dict[str, int]) -> list[RankedPage]:
    """Reject unverified positions/IDs from adapters instead of trusting ranks."""
    result: list[RankedPage] = []
    seen: set[str] = set()
    for i, row in enumerate(rows, 1):
        chunk_id = row.chunk_id
        if chunk_id in seen or expected_ids.get(chunk_id) != row.page_number:
            raise RetrievalError("Retrieved hit is duplicated or disagrees with saved chunk metadata")
        seen.add(chunk_id)
        result.append(RankedPage(rank=i, page_number=row.page_number, chunk_id=chunk_id))
    return result


def run_retrieval_evaluation(
    paper_id: str, *, data_dir: Path, suite_path: Path,
    k_values: str | Sequence[int] = DEFAULT_K_VALUES,
    dense_weight: float = 1.0, bm25_weight: float = 1.0, max_hits_per_page: int = 0,
    encoder_factory: Callable[[str, str], Encoder] | None = None,
    faiss_backend: Any | None = None,
) -> RetrievalEvaluationReport:
    paper_id = validate_paper_id(paper_id)
    ks = parse_k_values(k_values)
    dense_weight, bm25_weight = validate_rrf_weights(dense_weight, bm25_weight)
    validate_page_cap(max_hits_per_page)
    suite = load_retrieval_suite(Path(suite_path))
    if suite.paper_id != paper_id:
        raise RetrievalError(f"Suite paper_id {suite.paper_id} does not match requested paper_id {paper_id}")
    try:
        bundle = load_saved_embedding_bundle(paper_id, data_dir=data_dir)
    except (OSError, ValueError) as exc:
        raise RetrievalError(str(exc)) from exc
    chunks = bundle.document.chunks
    by_id = {chunk.chunk_id: chunk.page_number for chunk in chunks}
    if len(by_id) != len(chunks):
        raise RetrievalError("Duplicate saved chunk identifiers")
    present_pages = {chunk.page_number for chunk in chunks}
    for case in suite.cases:
        missing = {label.page_number for label in case.labels} - present_pages
        if missing:
            raise RetrievalError(f"Case {case.id} labels pages missing from saved chunks: {sorted(missing)}")
    lexical = BM25Index(chunks)
    # Loading sentence-transformers for each query/method is expensive. Reuse
    # the same pinned encoder for all queries without reusing any query vectors.
    encoder_cache: dict[tuple[str, str], Encoder] = {}

    def shared_encoder(name: str, revision: str) -> Encoder:
        key = (name, revision)
        if key not in encoder_cache:
            encoder_cache[key] = (encoder_factory or SentenceTransformerEncoder)(name, revision)
        return encoder_cache[key]

    max_k = ks[-1]
    cases: list[QueryEvaluation] = []
    for case in suite.cases:
        sparse = lexical.search(case.question, limit=max_k)
        bm25_rows = [
            RankedPage(rank=i, page_number=chunks[hit.position].page_number,
                       chunk_id=chunks[hit.position].chunk_id)
            for i, hit in enumerate(sparse, 1)
        ]
        dense = search_paper(
            paper_id, case.question, data_dir=data_dir, top_k=max_k,
            encoder_factory=shared_encoder, faiss_backend=faiss_backend,
        )
        fused = hybrid_search_paper(
            paper_id, case.question, data_dir=data_dir, top_k=max_k,
            context_window=0, encoder_factory=shared_encoder, faiss_backend=faiss_backend,
            dense_weight=dense_weight, bm25_weight=bm25_weight,
            max_hits_per_page=max_hits_per_page,
        )
        rows = {
            "bm25": _rank_pages(bm25_rows, expected_ids=by_id),
            "faiss": _rank_pages(dense.hits, expected_ids=by_id),
            "hybrid": _rank_pages(fused.hits, expected_ids=by_id),
        }
        relevant = {item.page_number for item in case.labels}
        cases.append(QueryEvaluation(
            id=case.id, question=case.question, relevant_pages=sorted(relevant),
            top_hits=rows,
            metrics={method: {str(k): score_at_k(hits, relevant, k) for k in ks}
                     for method, hits in rows.items()},
        ))
    macro: dict[str, dict[str, MetricSummary]] = {}
    for method in METHODS:
        macro[method] = {}
        for k in ks:
            results = [case.metrics[method][str(k)] for case in cases]
            n = len(results)
            macro[method][str(k)] = MetricSummary(**{
                field: round(sum(getattr(r, field) for r in results) / n, 6)
                for field in ("precision_at_k", "recall_at_k", "mrr_at_k", "hit_at_k")
            })
    return RetrievalEvaluationReport(
        suite_id=suite.suite_id, paper_id=paper_id,
        annotation_status=suite.annotation_status,
        corpus_chunk_sha256=bundle.chunk_file_sha256,
        vector_count=bundle.manifest.vector_count,
        embedding_model=bundle.manifest.model_name,
        model_revision=bundle.manifest.model_revision,
        query_count=len(cases), k_values=list(ks), methods=list(METHODS),
        macro_results=macro, query_results=cases,
        dense_weight=dense_weight, bm25_weight=bm25_weight, max_hits_per_page=max_hits_per_page,
    )


def render_markdown_report(report: RetrievalEvaluationReport) -> str:
    lines = [
        f"# PaperTrail — Retrieval Evaluation ({report.suite_id})",
        "",
        f"- Paper: `{report.paper_id}`",
        f"- Questions: **{report.query_count}**; annotation status: **{report.annotation_status}**",
        f"- Embedding model: `{report.embedding_model}` @ `{report.model_revision}`",
        f"- Chunk manifest SHA-256: `{report.corpus_chunk_sha256}`",
        "- Scoring only primary retrieved chunks (not neighbor context)",
        f"- Hybrid RRF weights: dense={report.dense_weight:g}, BM25={report.bm25_weight:g}; page cap={report.max_hits_per_page}",
        "",
        "## Macro-average results (page-proxy labels)",
        "",
        "| Method | K | Precision@K | Recall@K | MRR@K | Hit@K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for k in report.k_values:
            metric = report.macro_results[method][str(k)]
            lines.append(f"| {method} | {k} | {metric.precision_at_k:.3f} | "
                         f"{metric.recall_at_k:.3f} | {metric.mrr_at_k:.3f} | {metric.hit_at_k:.3f} |")
    lines.extend(["", "## Per-question page ranks", ""])
    for case in report.query_results:
        lines.extend([
            f"### {case.id}: {case.question}",
            f"- Labeled pages: {', '.join(str(p) for p in case.relevant_pages)}",
        ])
        for method in METHODS:
            pages = ", ".join(str(hit.page_number) for hit in case.top_hits[method]) or "(none)"
            lines.append(f"- {method} ranked pages: {pages}")
        lines.append("")
    lines.extend([
        "## Limitations", "",
        report.methodology_note,
        "",
        "The manually drafted page labels require PDF/source review before publication. "
        "One paper, a small query set, and potentially incomplete labels are insufficient "
        "to establish general retrieval superiority. Compare models on the same frozen corpus "
        "and suite, and report negative findings too.",
        "",
        "PDF extraction quality independently verified: **No**.", "",
    ])
    return "\n".join(lines)


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".tmp_eval_", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_retrieval_report(report: RetrievalEvaluationReport, *, data_dir: Path) -> tuple[Path, Path]:
    """Only generated aggregate reports; raw article text is never exported."""
    stem = f"{report.paper_id.replace('/', '__')}_{report.suite_id}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
        raise RetrievalError("Unsafe report filename")
    destination = Path(data_dir) / "evaluations"
    json_path = destination / f"{stem}.json"
    md_path = destination / f"{stem}.md"
    try:
        _write_atomic(json_path, report.model_dump_json(indent=2))
        _write_atomic(md_path, render_markdown_report(report))
    except OSError as exc:
        raise RetrievalError(f"Could not write evaluation report: {type(exc).__name__}") from exc
    return json_path, md_path

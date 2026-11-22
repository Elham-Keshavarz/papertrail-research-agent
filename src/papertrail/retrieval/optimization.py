"""Phase 2.7: held-out, non-deploying RRF weight diagnostics.

All judgments in the bundled suite are provisional PDF *page* labels. Search
candidates are collected once per question, then frozen before trial weights
are scored. Weight selection reads training labels only; validation is evaluated
only AFTER selection. No tuning result changes production defaults automatically.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from papertrail.embedding.encoder import Encoder, SentenceTransformerEncoder
from papertrail.embedding.pipeline import load_saved_embedding_bundle
from papertrail.ingestion.pdf_downloader import validate_paper_id
from .bm25 import BM25Index
from .faiss_store import RetrievalError
from .fusion import fuse_candidates, validate_rrf_weights, select_page_diverse
from .quality_eval import (RankedPage, load_retrieval_suite, score_at_k,
                           _write_atomic)
from .search import search_paper

# Explicit, fixed candidate grid: simple enough to audit. All scoring uses
# previously collected ranks, not repeated encoder calls or changing qrels.
WEIGHT_GRID = ((1., 0.), (0., 1.), (1., 1.), (1., 2.), (1., 4.), (2., 1.), (4., 1.))
# Page diversification is optional and scored separately, not secretly applied.
CANDIDATE_GRID = tuple((dw, bw, cap) for cap in (0, 2) for dw, bw in WEIGHT_GRID)
VALID_METRICS = ("mrr_at_k", "recall_at_k", "hit_at_k", "precision_at_k")
DEFAULT_SEED = 42


class WeightResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dense_weight: float
    bm25_weight: float
    page_cap: int = 0
    train_score: float = Field(ge=0, le=1)
    validation_score: float = Field(ge=0, le=1)


class ReviewCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    page_number: int = Field(ge=1)
    best_rank: int = Field(ge=1)
    retrieval_methods: list[str]
    note: str = "Unlabeled page candidate for MANUAL review; not automatically relevant or irrelevant"


class OptimizationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    suite_id: str
    suite_sha256: str
    corpus_chunk_sha256: str
    model_name: str
    model_revision: str
    vector_count: int = Field(gt=0)
    annotation_status: str
    seed: int
    train_query_ids: list[str]
    validation_query_ids: list[str]
    metric: str
    k: int
    grid_results: list[WeightResult]
    selected_dense_weight: float
    selected_bm25_weight: float
    selected_page_cap: int = 0
    baseline_validation_score: float
    selected_validation_score: float
    bm25_validation_score: float
    faiss_validation_score: float
    validation_delta_vs_default: float
    deployment_recommendation: str
    flagged_unlabeled_pages: list[ReviewCandidate]
    candidate_collection_count: int
    pdf_extraction_quality_verified: bool = False
    methodology_note: str = (
        "Exploratory page-proxy tuning on provisional, possibly incomplete judgments. "
        "Deterministic query-level train/holdout split; choose grid weights using TRAIN labels ONLY. "
        "One two-question holdout from one paper is NOT an independent multi-paper benchmark. "
        "Unlabeled pages are NOT assumed irrelevant outside this limited scoring proxy. "
        "No automatic deployment; PDF extraction/layout and qrels need manual source audit."
    )


def _score(records: Sequence[RankedPage], labels: set[int], *, metric: str, k: int) -> float:
    return float(getattr(score_at_k(records, labels, k), metric))


def _macro(
    rankings: dict[str, list[RankedPage]],
    label_sets: dict[str, set[int]], ids: Sequence[str], *, metric: str, k: int,
) -> float:
    return round(sum(_score(rankings[case_id], label_sets[case_id], metric=metric, k=k)
                     for case_id in ids) / len(ids), 6)


def _split_queries(ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    if len(ids) < 4:
        raise RetrievalError("Holdout tuning needs at least four labeled questions")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32-1:
        raise RetrievalError("seed must be an integer in 0..4294967295")
    shuffled = sorted(ids)  # input case order must not change split
    random.Random(seed).shuffle(shuffled)
    validation_n = max(2, math.ceil(len(ids) / 4))
    return sorted(shuffled[validation_n:]), sorted(shuffled[:validation_n])


def _unlabeled_page_candidates(
    *, case_id: str, labels: set[int], rankings: dict[str, list[RankedPage]],
    max_rank: int = 3,
) -> list[ReviewCandidate]:
    possibilities: dict[int, dict[str, Any]] = {}
    for method in ("bm25", "faiss", "hybrid_default"):
        for hit in rankings[method][:max_rank]:
            if hit.page_number in labels:
                continue
            slot = possibilities.setdefault(hit.page_number, {"ranks": [], "methods": []})
            slot["ranks"].append(hit.rank)
            slot["methods"].append(method)
    return [ReviewCandidate(case_id=case_id, page_number=page,
                            best_rank=min(info["ranks"]),
                            retrieval_methods=info["methods"])
            for page, info in sorted(possibilities.items(), key=lambda p: (min(p[1]["ranks"]), p[0]))]


def run_retrieval_optimization(
    paper_id: str, *, data_dir: Path, suite_path: Path, seed: int = DEFAULT_SEED,
    metric: str = "mrr_at_k", k: int = 5,
    encoder_factory: Callable[[str, str], Encoder] | None = None,
    faiss_backend: Any | None = None,
) -> OptimizationReport:
    paper_id = validate_paper_id(paper_id)
    if metric not in VALID_METRICS:
        raise RetrievalError(f"metric must be one of: {', '.join(VALID_METRICS)}")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
        raise RetrievalError("k must be an integer in 1..20")
    suite_path = Path(suite_path)
    suite = load_retrieval_suite(suite_path)
    if suite.paper_id != paper_id:
        raise RetrievalError("Optimization suite does not match requested paper_id")
    ids = [case.id for case in suite.cases]
    train_ids, validation_ids = _split_queries(ids, seed)
    try:
        bundle = load_saved_embedding_bundle(paper_id, data_dir=data_dir)
        suite_sha = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise RetrievalError(str(exc)) from exc
    chunks = bundle.document.chunks
    positions = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
    if len(positions) != len(chunks) or len(chunks) != bundle.manifest.vector_count:
        raise RetrievalError("Duplicate or misaligned saved chunks")
    page_set = {chunk.page_number for chunk in chunks}
    labels = {case.id: {r.page_number for r in case.labels} for case in suite.cases}
    for case_id, pages in labels.items():
        missing = pages - page_set
        if missing:
            raise RetrievalError(f"Case {case_id} labels pages absent from the corpus: {sorted(missing)}")

    sparse_index = BM25Index(chunks)
    encoder_cache: dict[tuple[str, str], Encoder] = {}

    def shared_encoder(name: str, revision: str) -> Encoder:
        key = (name, revision)
        if key not in encoder_cache:
            encoder_cache[key] = (encoder_factory or SentenceTransformerEncoder)(name, revision)
        return encoder_cache[key]

    def to_rows(items: Sequence[int]) -> list[RankedPage]:
        return [RankedPage(rank=rank, page_number=chunks[pos].page_number,
                           chunk_id=chunks[pos].chunk_id)
                for rank, pos in enumerate(items, 1)]

    # First collect every query's independent source rankings once. No labels
    # influence this phase; changing a weight never triggers another retrieval.
    candidates: dict[str, tuple[Any, Any]] = {}
    base_rankings: dict[str, dict[str, list[RankedPage]]] = {}
    for case in suite.cases:
        dense = search_paper(paper_id, case.question, data_dir=data_dir,
                             top_k=min(20, len(chunks)), encoder_factory=shared_encoder,
                             faiss_backend=faiss_backend)
        sparse = sparse_index.search(case.question, limit=min(60, len(chunks)))
        # Use the same verified chunk positions for every method and weight.
        if len({h.chunk_id for h in dense.hits}) != len(dense.hits) or any(
            h.chunk_id not in positions or chunks[positions[h.chunk_id]].page_number != h.page_number
            for h in dense.hits
        ):
            raise RetrievalError("Dense candidate not aligned with verified corpus")
        candidates[case.id] = (dense.hits, sparse)
        default = fuse_candidates(dense.hits, sparse, positions, chunk_count=len(chunks))
        base_rankings[case.id] = {
            "bm25": to_rows([h.position for h in sparse[:k]]),
            "faiss": to_rows([positions[h.chunk_id] for h in dense.hits[:k]]),
            "hybrid_default": to_rows([h.position for h in default[:k]]),
        }

    pages_by_position = [chunk.page_number for chunk in chunks]
    grid_rankings: dict[tuple[float, float, int], dict[str, list[RankedPage]]] = {}
    for dense_weight, bm25_weight, page_cap in CANDIDATE_GRID:
        validate_rrf_weights(dense_weight, bm25_weight)
        grid_rankings[(dense_weight, bm25_weight, page_cap)] = {
            case_id: to_rows([
                hit.position for hit in select_page_diverse(
                    fuse_candidates(dense, sparse, positions, chunk_count=len(chunks),
                        dense_weight=dense_weight, bm25_weight=bm25_weight),
                    pages_by_position, top_k=k, max_hits_per_page=page_cap)
            ])
            for case_id, (dense, sparse) in candidates.items()
        }

    # Selection uses train labels only, not validation score.
    train_scores = {weights: _macro(ranks, labels, train_ids, metric=metric, k=k)
                    for weights, ranks in grid_rankings.items()}
    chosen = max(CANDIDATE_GRID, key=lambda weights: (
        train_scores[weights], int(weights == (1.0, 1.0, 0)), -weights[2],
        -abs(weights[0] - weights[1]), -max(weights), -weights[0],
    ))
    records = [WeightResult(
        dense_weight=dw, bm25_weight=bw, page_cap=cap,
        train_score=train_scores[(dw, bw, cap)],
        validation_score=_macro(grid_rankings[(dw, bw, cap)], labels, validation_ids, metric=metric, k=k),
    ) for dw, bw, cap in CANDIDATE_GRID]
    by_setting = {(r.dense_weight, r.bm25_weight, r.page_cap): r for r in records}
    selected_val = by_setting[chosen].validation_score
    baseline_val = by_setting[(1., 1., 0)].validation_score
    lexical_val = by_setting[(0., 1., 0)].validation_score
    dense_val = by_setting[(1., 0., 0)].validation_score
    # Guardrail: never deploy on this toy benchmark, even if scores rise.
    if selected_val > baseline_val + 1e-6:
        recommendation = "exploratory_gain_only_manual_review_required_no_auto_deploy"
    elif selected_val < baseline_val - 1e-6:
        recommendation = "holdout_regression_keep_default_and_review_labels"
    else:
        recommendation = "no_holdout_gain_keep_default"
    flags = [flag for case in suite.cases for flag in _unlabeled_page_candidates(
        case_id=case.id, labels=labels[case.id], rankings=base_rankings[case.id])]
    return OptimizationReport(
        paper_id=paper_id, suite_id=suite.suite_id,
        suite_sha256=suite_sha, corpus_chunk_sha256=bundle.chunk_file_sha256,
        model_name=bundle.manifest.model_name, model_revision=bundle.manifest.model_revision,
        vector_count=len(chunks), annotation_status=suite.annotation_status,
        seed=seed, train_query_ids=train_ids, validation_query_ids=validation_ids,
        metric=metric, k=k, grid_results=records,
        selected_dense_weight=chosen[0], selected_bm25_weight=chosen[1], selected_page_cap=chosen[2],
        baseline_validation_score=baseline_val, selected_validation_score=selected_val,
        bm25_validation_score=lexical_val, faiss_validation_score=dense_val,
        validation_delta_vs_default=round(selected_val - baseline_val, 6),
        deployment_recommendation=recommendation,
        flagged_unlabeled_pages=flags, candidate_collection_count=len(candidates),
    )


def render_optimization_markdown(report: OptimizationReport) -> str:
    lines = [
        f"# PaperTrail — Held-out RRF Optimization ({report.suite_id})", "",
        f"- Paper: `{report.paper_id}`", f"- Annotation status: **{report.annotation_status}**",
        f"- Corpus chunks: `{report.vector_count}`", f"- Corpus SHA-256: `{report.corpus_chunk_sha256}`",
        f"- Suite SHA-256: `{report.suite_sha256}`", f"- Embedding: `{report.model_name}` @ `{report.model_revision}`",
        f"- Fixed split seed: `{report.seed}`", f"- Training question IDs: {', '.join(report.train_query_ids)}",
        f"- Held-out question IDs: {', '.join(report.validation_query_ids)}",
        f"- Selection objective: **{report.metric}@{report.k}** (page-proxy)", "",
        "## All weights (train selects, holdout diagnoses)", "",
        "| Dense weight | BM25 weight | Page cap | Train | Holdout |", "|---:|---:|---:|---:|---:|",
    ]
    for score in report.grid_results:
        lines.append(f"| {score.dense_weight:g} | {score.bm25_weight:g} | {score.page_cap} | "
                     f"{score.train_score:.3f} | {score.validation_score:.3f} |")
    lines.extend([
        "", f"- **Selected from train only:** dense={report.selected_dense_weight:g}, "
        f"BM25={report.selected_bm25_weight:g}, page cap={report.selected_page_cap}",
        f"- Default RRF holdout: **{report.baseline_validation_score:.3f}**",
        f"- Selected RRF holdout: **{report.selected_validation_score:.3f}**",
        f"- BM25-only holdout: **{report.bm25_validation_score:.3f}**",
        f"- FAISS-only holdout: **{report.faiss_validation_score:.3f}**",
        f"- Selected vs default (holdout): **{report.validation_delta_vs_default:+.3f}**",
        f"- Recommendation: `{report.deployment_recommendation}`", "",
        "## Possible missing page labels — manual PDF review only", "",
        "| Case | Unlabeled page | Best rank | Retrieved by |",
        "|---|---:|---:|---|",
    ])
    for flag in report.flagged_unlabeled_pages:
        lines.append(f"| {flag.case_id} | {flag.page_number} | {flag.best_rank} | "
                     f"{', '.join(flag.retrieval_methods)} |")
    if not report.flagged_unlabeled_pages:
        lines.append("| (none) | | | |")
    lines.extend(["", "## Validity limits", "", report.methodology_note, "",
                  "The PDF and page labels have NOT been independently validated. "
                  "Do not publish holdout metrics as independent accuracy. "
                  "Do not change production weights automatically. "
                  "Label audits require actual reading of source pages.", ""])
    return "\n".join(lines)


def save_optimization_report(report: OptimizationReport, *, data_dir: Path) -> tuple[Path, Path]:
    stem = f"{report.paper_id.replace('/', '__')}_{report.suite_id}_rrf_optimization"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
        raise RetrievalError("Unsafe optimization report filename")
    root = Path(data_dir) / "evaluations"
    jpath, mpath = root / f"{stem}.json", root / f"{stem}.md"
    try:
        _write_atomic(jpath, report.model_dump_json(indent=2))
        _write_atomic(mpath, render_optimization_markdown(report))
    except OSError as exc:
        raise RetrievalError(f"Could not save optimization report: {type(exc).__name__}") from exc
    return jpath, mpath

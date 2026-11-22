"""Phase 3.3: bounded, auditable execution of existing real research tools.

The graph carries state and decides routing; this adapter invokes the exact
Phase 1/2 pipelines rather than fabricating evidence or generating an answer.
No LLM is required. arXiv network access requires explicit opt-in.
"""
from __future__ import annotations

import hashlib
import operator
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, Literal, Protocol, TypedDict

from pydantic import ValidationError

from papertrail.agent.workflow import AgentDependencyError, plan_research_node, validate_request_node
from papertrail.agent.assessment import assess_evidence, refinement_query
from papertrail.agent.faithfulness import FaithfulnessError, audit_checked_answer
from papertrail.agent.progress import research_checkpoint
from papertrail.agent.synthesis import (SynthesisError, build_evidence_register,
    make_synthesis_payload, validate_with_partial_recovery)
from papertrail.embedding.pipeline import load_saved_embedding_bundle
from papertrail.ingestion.pdf_downloader import validate_paper_id
from papertrail.models import ResearchRequest


@contextmanager
def _offline_model_guard(enabled: bool):
    """Disable network-based Hugging Face model resolution in offline tool mode.

    The guard is scoped to one synchronous graph invocation. It does not
    override a user-supplied HF cache or download models in the background.
    """
    if not enabled:
        yield
        return
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
    previous = {key: os.environ.get(key) for key in names}
    try:
        for key in names:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class ExecutionError(ValueError):
    """Invalid request or bounded tool configuration."""


@dataclass(frozen=True)
class ExecutionOptions:
    data_dir: Path = Path("data")
    allow_network: bool = False
    auto_select_first: bool = False
    top_k: int = 3
    query_style: Literal["baseline", "focused", "multi", "budgeted"] = "baseline"
    context_window: int = 1
    max_tool_calls: int = 12
    max_papers: int = 5
    max_pdf_pages: int = 100
    max_retries: int = 2
    timeout_seconds: float = 20.0
    chunk_size: int = 400
    overlap: int = 60
    max_refinement_rounds: int = 1
    min_hits_per_step: int = 2

    def __post_init__(self) -> None:
        if self.query_style not in {"baseline", "focused", "multi", "budgeted"}:
            raise ExecutionError("query_style must be baseline, focused, multi or budgeted")
        if self.query_style in {"multi", "budgeted"} and self.allow_network:
            raise ExecutionError("Multi-query experimental fusion is local-only")
        for name, value, lo, hi in (
            ("top_k", self.top_k, 1, 10),
            ("context_window", self.context_window, 0, 2),
            ("max_tool_calls", self.max_tool_calls, 1, 20),
            ("max_papers", self.max_papers, 1, 20),
            ("max_pdf_pages", self.max_pdf_pages, 1, 200),
            ("max_retries", self.max_retries, 0, 5),
            ("max_refinement_rounds", self.max_refinement_rounds, 0, 1),
            ("min_hits_per_step", self.min_hits_per_step, 1, 10),
        ):
            if type(value) is not int or not lo <= value <= hi:
                raise ExecutionError(f"{name} must be an integer between {lo} and {hi}")
        if not 1 <= self.timeout_seconds <= 120:
            raise ExecutionError("timeout_seconds must be between 1 and 120")
        if self.auto_select_first and not self.allow_network:
            raise ExecutionError("auto_select_first requires allow_network")
        from papertrail.chunking.chunker import validate_chunk_settings
        validate_chunk_settings(self.chunk_size, self.overlap)


class ResearchTools(Protocol):
    def discover(self, question: str, options: ExecutionOptions) -> list[dict]: ...
    def ingest(self, paper_id: str, options: ExecutionOptions) -> dict: ...
    def chunk(self, paper_id: str, options: ExecutionOptions) -> dict: ...
    def embed(self, paper_id: str, options: ExecutionOptions) -> dict: ...
    def index(self, paper_id: str, options: ExecutionOptions) -> dict: ...
    def retrieve(self, paper_id: str, query: str, options: ExecutionOptions) -> list[dict]: ...


class PaperTrailTools:
    """Production adapter: delegates to previously validated PaperTrail tools."""

    def discover(self, question: str, options: ExecutionOptions) -> list[dict]:
        if not options.allow_network:
            raise ExecutionError("Network access was not enabled")
        from papertrail.discovery.arxiv_client import ArxivClient
        # Keep arXiv search bounded and avoid querying the entire question as an
        # unlikely-to-match literal phrase. Only controlled alphanumeric keywords
        # enter the API expression, never user-supplied field syntax or commands.
        stopwords = {"what", "which", "where", "when", "are", "the", "these", "those",
                     "recent", "new", "paper", "papers", "methods", "method", "for",
                     "with", "about", "explain", "compare", "comparison", "their",
                     "between", "does", "did", "how", "and", "has", "have", "using",
                     "from", "that", "this", "into", "were", "versus", "traditional"}
        terms = list(dict.fromkeys(t for t in re.findall(r"[a-z][a-z0-9]*", question.lower())
                                   if len(t) > 2 and t not in stopwords))[:4]
        if not terms:
            raise ExecutionError("No safe arXiv search keywords could be extracted")
        expression = " AND ".join(f"all:{term}" for term in terms)
        with ArxivClient(timeout_seconds=options.timeout_seconds, max_retries=options.max_retries) as client:
            result = client.search(expression, max_results=min(options.max_papers, 5))
        return [
            {"paper_id": validate_paper_id(p.paper_id), "title": p.title[:300], "url": str(p.url)}
            for p in result.papers
        ]

    def ingest(self, paper_id: str, options: ExecutionOptions) -> dict:
        from papertrail.ingestion.pdf_downloader import ArxivPdfDownloader
        from papertrail.ingestion.pipeline import ingest_arxiv_pdf
        pdf = _paths(options.data_dir, paper_id)["pdf"]
        if not options.allow_network and not pdf.is_file():
            raise ExecutionError("Missing local PDF and arXiv downloading is disabled")
        # The real ingestion pipeline uses cached PDF bytes when present. In offline
        # mode, the downloader can never fetch even if the file vanishes mid-call.
        with ArxivPdfDownloader(timeout_seconds=options.timeout_seconds, max_retries=options.max_retries) as client:
            if not options.allow_network:
                client.fetch = lambda *_args: (_ for _ in ()).throw(ExecutionError("Offline mode prohibits PDF download"))
            document, _, _, cached = ingest_arxiv_pdf(
                paper_id, data_dir=options.data_dir, client=client,
                max_pages=options.max_pdf_pages, layout="auto",
            )
        return {"cached_pdf": cached, "pages_extracted": document.extracted_pages,
                "source_sha256": document.sha256}

    def chunk(self, paper_id: str, options: ExecutionOptions) -> dict:
        from papertrail.chunking.pipeline import chunk_saved_document
        doc, _ = chunk_saved_document(paper_id, data_dir=options.data_dir,
                                      chunk_size=options.chunk_size, overlap=options.overlap)
        return {"chunks": doc.total_chunks}

    def embed(self, paper_id: str, options: ExecutionOptions) -> dict:
        from papertrail.embedding.pipeline import embed_saved_document
        run = embed_saved_document(paper_id, data_dir=options.data_dir, batch_size=16)
        return {"cached": run.cached, "vectors": run.manifest.vector_count,
                "model": run.manifest.model_name}

    def index(self, paper_id: str, options: ExecutionOptions) -> dict:
        from papertrail.retrieval.faiss_store import build_paper_index
        # Validates every chunk/vector hash and the matching index manifest;
        # a missing/corrupt *local cache* may be rebuilt from verified vectors.
        report = build_paper_index(paper_id, data_dir=options.data_dir)
        return {"cached": report.cached, "vectors": report.manifest.vector_count,
                "index_type": report.manifest.index_type}

    def retrieve(self, paper_id: str, query: str, options: ExecutionOptions) -> list[dict]:
        from papertrail.retrieval.hybrid import hybrid_search_paper
        result = hybrid_search_paper(paper_id, query, data_dir=options.data_dir,
                                     top_k=options.top_k, context_window=options.context_window)
        return [
            {"chunk_id": h.chunk_id, "page_number": h.page_number, "citation": h.citation,
             "rank": h.rank, "rrf_score": h.score, "text": h.text,
             "context": [{"chunk_id": c.chunk_id, "citation": c.citation,
                          "page_number": c.page_number, "text": c.text}
                         for c in h.context]}
            for h in result.hits
        ]


def _paths(data_dir: Path, paper_id: str) -> dict[str, Path]:
    stem = validate_paper_id(paper_id).replace("/", "__")
    return {
        "pdf": data_dir / "papers" / f"{stem}.pdf",
        "extracted": data_dir / "extracted" / f"{stem}.json",
        "chunks": data_dir / "chunks" / f"{stem}.json",
        "vectors": data_dir / "embeddings" / f"{stem}.npy",
        "embedding_manifest": data_dir / "embeddings" / f"{stem}.manifest.json",
        "index": data_dir / "indexes" / f"{stem}.faiss",
        "index_manifest": data_dir / "indexes" / f"{stem}.manifest.json",
    }


def _source_audit(paper_id: str, data_dir: Path) -> dict:
    """Optional extra source-chain check, never an answer-factuality judgment."""
    from papertrail.ingestion.pdf_extractor import ExtractedDocument
    paths = _paths(data_dir, paper_id)
    pdf, extracted = paths["pdf"], paths["extracted"]
    if not pdf.is_file() or not extracted.is_file():
        return {"status": "source_files_not_both_available", "pdf_extraction_quality_verified": False}
    from papertrail.ingestion.pdf_downloader import MAX_PDF_BYTES
    from papertrail.chunking.pipeline import MAX_EXTRACTED_JSON_BYTES
    if pdf.stat().st_size > MAX_PDF_BYTES or extracted.stat().st_size > MAX_EXTRACTED_JSON_BYTES:
        raise ExecutionError("Source PDF or extracted JSON exceeds size limits")
    doc = ExtractedDocument.model_validate_json(extracted.read_bytes())
    if doc.paper_id != paper_id or hashlib.sha256(pdf.read_bytes()).hexdigest() != doc.sha256:
        raise ExecutionError("PDF and extracted document provenance hashes do not match")
    from papertrail.chunking.chunker import ChunkedDocument
    from papertrail.embedding.pipeline import MAX_CHUNK_FILE_BYTES
    chunks_path = paths["chunks"]
    if chunks_path.is_file():
        if chunks_path.stat().st_size > MAX_CHUNK_FILE_BYTES:
            raise ExecutionError("Chunk cache exceeds size limit")
        chunks = ChunkedDocument.model_validate_json(chunks_path.read_bytes())
        if chunks.paper_id != paper_id or chunks.source_sha256 != doc.sha256:
            raise ExecutionError("Chunks do not match extracted PDF source")
        text_by_page = {page.page_number: page.text for page in doc.pages}
        if len(text_by_page) != len(doc.pages):
            raise ExecutionError("Extracted PDF has duplicate page identifiers")
        for chunk in chunks.chunks:
            text = text_by_page.get(chunk.page_number)
            if text is None or text[chunk.start_char:chunk.end_char] != chunk.text:
                raise ExecutionError("Chunk text offsets do not match extracted PDF page")
    # The extraction algorithm and qrels still require human/source review.
    return {"status": "pdf_extracted_hash_matched", "pdf_extraction_quality_verified": False}


class ToolEvent(TypedDict):
    name: str
    status: str
    detail: dict


class ExecutionState(TypedDict, total=False):
    question: str
    paper_id: str | None
    max_papers: int
    planner_requested: str
    status: str
    plan: list[dict]
    question_analysis: dict
    plan_validation: str
    planner_used: str
    planner_fallback_reason: str | None
    llm_calls: int
    discovery: list[dict]
    source_audit: dict
    tool_calls: int
    tool_events: list[ToolEvent]
    evidence: list[dict]
    retrieval_cursor: int
    assessment: dict
    retry_attempts: dict[str, int]
    retry_justification: list[dict]
    synthesis: dict
    synthesis_calls: int
    errors: list[str]
    trace: Annotated[list[dict], operator.add]


def _event(name: str, status: str, **detail: object) -> ToolEvent:
    return {"name": name, "status": status, "detail": detail}


def _invoke_tool(state: ExecutionState, options: ExecutionOptions, name: str, invoke):
    count = state.get("tool_calls", 0)
    if count >= options.max_tool_calls:
        raise ExecutionError("tool_call_budget_exhausted")
    # Count failures too: any attempted tool invocation consumes the budget.
    try:
        value = invoke()
    except Exception as exc:
        # Avoid leaking raw provider responses, auth headers or filesystem paths.
        return count + 1, None, type(exc).__name__
    return count + 1, value, None


def _fail(stage: str, *, count: int, events: list[ToolEvent], exception: str) -> dict:
    return {"status": "tool_failed", "tool_calls": count,
            "tool_events": events + [_event(stage, "failed", error_type=exception)],
            "errors": [f"{stage} failed ({exception})"],
            "trace": [{"node": stage, "event": "tool_failed"}]}


def build_tool_graph(options: ExecutionOptions, tools: ResearchTools | None = None,
                     planner_provider=None, *, assess_mode: bool = False,
                     synthesize_mode: bool = False, synthesis_provider: Callable[[dict], dict] | None = None):
    """Compile the *real* bounded LangGraph tool workflow; no answer generator."""
    try:
        from langgraph.graph import START, END, StateGraph
    except ImportError as exc:
        raise AgentDependencyError('LangGraph missing. Install project with the "agent" extra') from exc
    adapter = tools if tools is not None else PaperTrailTools()

    def choose_source(state: ExecutionState) -> dict:
        if state.get("paper_id") is not None:
            return {"status": "source_selected", "trace": [{"node": "select_source", "event": "user_paper_id"}]}
        if not options.allow_network:
            return {"status": "needs_paper_selection", "trace": [{"node": "select_source", "event": "offline_no_paper_id"}]}
        return {"status": "discover_requested", "trace": [{"node": "select_source", "event": "network_discovery_opted_in"}]}

    def do_discovery(state: ExecutionState) -> dict:
        try:
            count, candidates, failure = _invoke_tool(state, options, "arxiv_search",
                                                       lambda: adapter.discover(state["question"], options))
        except ExecutionError:
            return {"status": "tool_budget_exhausted", "errors": ["Tool budget exhausted"],
                    "trace": [{"node": "discover", "event": "budget_exhausted"}]}
        if failure:
            return _fail("arxiv_search", count=count, events=state.get("tool_events", []), exception=failure)
        try:
            if not isinstance(candidates, list) or len(candidates) > min(options.max_papers, 5):
                raise ValueError("Unbounded or malformed search response")
            cleaned = []
            for item in candidates:
                if not isinstance(item, dict):
                    raise ValueError("Bad search result")
                identifier = validate_paper_id(item["paper_id"])
                cleaned.append({"paper_id": identifier, "title": str(item["title"])[:300],
                                "url": f"https://arxiv.org/abs/{identifier}"})
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return _fail("arxiv_search_validation", count=count,
                         events=state.get("tool_events", []), exception=type(exc).__name__)
        updates: dict = {"tool_calls": count, "discovery": cleaned,
                         "tool_events": state.get("tool_events", []) + [_event("arxiv_search", "success", results=len(cleaned))]}
        if not cleaned:
            updates["status"] = "no_papers_found"
            updates["trace"] = [{"node": "discover", "event": "no_results"}]
        elif not options.auto_select_first:
            updates["status"] = "paper_selection_required"
            updates["trace"] = [{"node": "discover", "event": "candidates_require_review"}]
        else:
            updates["paper_id"] = cleaned[0]["paper_id"]
            updates["status"] = "source_selected"
            updates["trace"] = [{"node": "discover", "event": "first_candidate_selected_unverified"}]
        return updates

    def prepare(state: ExecutionState) -> dict:
        paper_id = validate_paper_id(state["paper_id"])
        paths = _paths(options.data_dir, paper_id)
        events = list(state.get("tool_events", []))
        count = state.get("tool_calls", 0)
        for step, missing, call in (
            ("pdf_ingestion", not paths["extracted"].is_file() and not paths["chunks"].is_file(),
             lambda: adapter.ingest(paper_id, options)),
            ("chunk_document", not paths["chunks"].is_file(), lambda: adapter.chunk(paper_id, options)),
            ("embed_document", not (paths["vectors"].is_file() and paths["embedding_manifest"].is_file()),
             lambda: adapter.embed(paper_id, options)),
            ("index_document", True, lambda: adapter.index(paper_id, options)),
        ):
            if not missing:
                if step == "pdf_ingestion" and not paths["extracted"].is_file():
                    events.append(_event(step, "skipped_using_downstream_chunks",
                                         source_extracted_json_missing=True))
                else:
                    events.append(_event(step, "reused_existing_artifact"))
                continue
            if step == "pdf_ingestion" and not options.allow_network and not paths["pdf"].is_file():
                return {"status": "offline_source_missing", "tool_calls": count,
                        "tool_events": events, "errors": ["No local PDF/extracted text; use --allow-network for arXiv download"],
                        "trace": [{"node": "prepare", "event": "offline_missing_source"}]}
            try:
                count, result, failure = _invoke_tool({"tool_calls": count}, options, step, call)
            except ExecutionError:
                return {"status": "tool_budget_exhausted", "tool_calls": count, "tool_events": events,
                        "errors": ["Tool budget exhausted"],
                        "trace": [{"node": "prepare", "event": "budget_exhausted"}]}
            if failure:
                return _fail(step, count=count, events=events, exception=failure)
            events.append(_event(step, "success", **(result if isinstance(result, dict) else {})))
        # Fail closed for mismatched PDF/extracted provenance if both are available.
        # A fake adapter can supply fixture artifacts; the source check never
        # claims manual extraction quality or relevance verification.
        try:
            source_audit = _source_audit(paper_id, options.data_dir)
        except (OSError, ValueError, ValidationError) as exc:
            return _fail("source_provenance", count=count, events=events, exception=type(exc).__name__)
        return {"status": "prepared_integrity_checked", "source_audit": source_audit,
                "tool_calls": count, "tool_events": events,
                "trace": [{"node": "prepare", "event": "cached_or_built_artifacts"}]}

    def retrieve_one(state: ExecutionState) -> dict:
        step_list = [p for p in state.get("plan", []) if p.get("proposed_tool") == "hybrid_retrieval"]
        cursor = state.get("retrieval_cursor", 0)
        if cursor >= len(step_list):
            return {"status": "retrieval_complete" if (assess_mode or synthesize_mode) else "evidence_retrieved_no_answer",
                    "trace": [{"node": "retrieve", "event": "retrieval_complete"}]}
        task = step_list[cursor]
        question = task["search_query"]
        count = state.get("tool_calls", 0)
        events = list(state.get("tool_events", []))
        query_traces = []
        if options.query_style in {"multi", "budgeted"}:
            from papertrail.agent.query_rewriting import focused_queries
            intents = state["question_analysis"]["intents"]
            specialized = focused_queries(state["question"], intents)
            if len(specialized) != len(step_list):
                return {"status": "query_plan_invalid", "errors": ["Facet/query count mismatch"],
                        "trace": [{"node": "retrieve", "event": "query_plan_invalid"}]}
            query_variants = [("baseline", question), ("focused", specialized[cursor])]
        else:
            query_variants = [(options.query_style, question)]

        branch_results = {}
        for style, query_text in query_variants:
            try:
                count, passages, failure = _invoke_tool({"tool_calls": count}, options, "hybrid_retrieval",
                    lambda q=query_text: adapter.retrieve(state["paper_id"], q, options))
            except ExecutionError:
                return {"status": "tool_budget_exhausted", "tool_calls": count, "tool_events": events,
                        "errors": ["Tool budget exhausted"],
                        "trace": [{"node": "retrieve", "event": "budget_exhausted"}]}
            if failure:
                return _fail("hybrid_retrieval", count=count, events=events, exception=failure)
            try:
                approved = _validate_evidence_passages(passages, paper_id=state["paper_id"], top_k=options.top_k)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                return _fail("evidence_validation", count=count, events=events, exception=type(exc).__name__)
            branch_results[style] = approved
            query_traces.append({"style": style, "query": query_text,
                                 "chunk_ids": [hit["chunk_id"] for hit in approved]})
            events.append(_event("hybrid_retrieval", "success", step_id=task["id"],
                                 style=style, hits=len(approved)))

        if options.query_style in {"multi", "budgeted"}:
            from papertrail.agent.multi_query import fuse_validated_queries, fuse_budgeted_queries
            try:
                fusion = fuse_budgeted_queries if options.query_style == "budgeted" else fuse_validated_queries
                approved = fusion(branch_results, limit=options.top_k)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                return _fail("multi_query_fusion", count=count, events=events, exception=type(exc).__name__)
        else:
            approved = branch_results[options.query_style]
        evidence = list(state.get("evidence", [])) + [{
            "step_id": task["id"], "query": question, "hits": approved,
            "query_traces": query_traces,
        }]
        return {"tool_calls": count, "tool_events": events,
                "evidence": evidence, "retrieval_cursor": cursor + 1,
                "status": "retrieving" if cursor + 1 < len(step_list) else (
                    "retrieval_complete" if (assess_mode or synthesize_mode) else (
                        "evidence_retrieved_no_answer" if any(e["hits"] for e in evidence) else "no_retrieval_hits"
                    )
                ),
                "trace": [{"node": "retrieve", "event": f"retrieved_{cursor+1}_of_{len(step_list)}"}]}

    def assess_node(state: ExecutionState) -> dict:
        report = assess_evidence(
            state.get("plan", []), state.get("evidence", []),
            min_hits_per_step=options.min_hits_per_step, top_k=options.top_k,
            source_audit=state.get("source_audit"), retry_attempts=state.get("retry_attempts"),
        )
        if report["decision"] == "structurally_covered_unverified":
            status = "evidence_assessed_unverified"
        elif report["decision"] == "no_retrieval_steps":
            status = "no_retrieval_steps"
        else:
            status = "evidence_gaps_detected"
        return {"assessment": report, "status": status,
                "trace": [{"node": "assess_evidence", "event": report["decision"]}]}

    def should_retry(state: ExecutionState) -> str:
        if state.get("status") != "evidence_gaps_detected":
            return "finalize"
        attempts = state.get("retry_attempts", {})
        if any(attempts.get(step_id, 0) < options.max_refinement_rounds
               for step_id in state.get("assessment", {}).get("gap_step_ids", [])):
            return "retry"
        return "finalize"

    def retry_one(state: ExecutionState) -> dict:
        attempts = dict(state.get("retry_attempts", {}))
        gaps = state.get("assessment", {}).get("gap_step_ids", [])
        step_id = next((s for s in gaps if attempts.get(s, 0) < options.max_refinement_rounds), None)
        if step_id is None:
            return {"status": "retry_complete", "trace": [{"node": "retry_evidence", "event": "no_pending_gaps"}]}
        step = next(p for p in state["plan"] if p["id"] == step_id)
        attempts[step_id] = attempts.get(step_id, 0) + 1
        # A later retry would repeat the same simplified query, which is not useful:
        # one attempt at most until a genuinely distinct query strategy is implemented.
        query = refinement_query(original_question=state["question"], planned_query=step["search_query"])
        if query is None:
            return {"retry_attempts": attempts, "status": "retry_complete",
                    "trace": [{"node": "retry_evidence", "event": "no_distinct_safe_query"}]}
        events = list(state.get("tool_events", []))
        count = state.get("tool_calls", 0)
        try:
            count, passages, failure = _invoke_tool(state, options, "hybrid_retrieval_retry",
                lambda: adapter.retrieve(state["paper_id"], query, options))
        except ExecutionError:
            return {"status": "tool_budget_exhausted", "retry_attempts": attempts,
                    "errors": ["Tool budget exhausted before evidence retry"],
                    "trace": [{"node": "retry_evidence", "event": "budget_exhausted"}]}
        if failure:
            return {**_fail("hybrid_retrieval_retry", count=count, events=events, exception=failure),
                    "retry_attempts": attempts}
        try:
            additional = _validate_evidence_passages(passages, paper_id=state["paper_id"], top_k=options.top_k)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return {**_fail("retry_evidence_validation", count=count, events=events,
                            exception=type(exc).__name__), "retry_attempts": attempts}
        evidence = [dict(entry) for entry in state.get("evidence", [])]
        for entry in evidence:
            if entry["step_id"] == step_id:
                known = {p["chunk_id"] for p in entry["hits"]}
                novel = [p for p in additional if p["chunk_id"] not in known]
                # Limit total per-facet source text even after retries.
                entry["hits"] = entry["hits"] + novel[:max(0, options.top_k * 2 - len(entry["hits"]))]
                break
        else:
            evidence.append({"step_id": step_id, "query": step["search_query"], "hits": additional})
        return {
            "retry_attempts": attempts, "evidence": evidence, "tool_calls": count,
            "retry_justification": state.get("retry_justification", []) + [{
                "step_id": step_id, "reason": "below_structural_evidence_threshold",
                "strategy": "original_user_question_only", "new_query": query,
                "attempted": True,
            }],
            "tool_events": events + [_event("hybrid_retrieval_retry", "success", step_id=step_id, hits=len(additional))],
            "status": "retry_complete",
            "trace": [{"node": "retry_evidence", "event": f"retry_{step_id}"}],
        }

    def synthesize_node(state: ExecutionState) -> dict:
        # Structural coverage is a prerequisite, never a factual-sufficiency claim.
        report = state.get("assessment") or {}
        if report.get("decision") != "structurally_covered_unverified":
            return {"status": "insufficient_evidence", "synthesis": {
                "status": "insufficient_evidence", "answer": None, "claims": [],
                "missing_step_ids": report.get("gap_step_ids", []),
                "reason": "structural_evidence_gaps", "human_review_required": True,
            }, "trace": [{"node": "synthesize_answer", "event": "abstained_on_structural_gaps"}]}
        try:
            register = build_evidence_register(state.get("plan", []), state.get("evidence", []),
                                               state["paper_id"])
            payload = make_synthesis_payload(state["question"], state.get("plan", []), register)
        except (ValueError, TypeError, KeyError, SynthesisError):
            return {"status": "citation_validation_failed", "synthesis": {
                "status": "citation_validation_failed", "answer": None, "claims": [],
                "reason": "source_register_rejected", "human_review_required": True,
            }, "trace": [{"node": "synthesize_answer", "event": "source_register_rejected"}]}
        public_register = [{k: r[k] for k in ('evidence_id', 'step_id', 'chunk_id', 'page_number', 'citation')}
                           for r in register]
        if synthesis_provider is None:
            return {"status": "ready_for_llm_synthesis", "synthesis": {
                "status": "ready_for_llm_synthesis", "answer": None, "claims": [],
                "source_register": public_register,
                "human_review_required": True,
                "note": "Preview only: no model call and no generated answer",
            }, "trace": [{"node": "synthesize_answer", "event": "preview_no_model_call"}]}
        # Exactly one remote generation attempt. Never log raw provider output.
        try:
            draft = synthesis_provider(payload)
        except Exception:
            return {"status": "llm_unavailable", "synthesis_calls": 1,
                    "synthesis": {"status": "llm_unavailable", "answer": None,
                                  "claims": [], "human_review_required": True},
                    "trace": [{"node": "synthesize_answer", "event": "model_request_failed"}]}
        try:
            checked = validate_with_partial_recovery(draft, state.get("plan", []), register)
        except (SynthesisError, ValueError, TypeError, KeyError):
            # Unanticipated validator errors remain fail-closed, without raw data.
            checked = {"reviewed": None, "rejections": [{"reason_code": "validation_rejected"}]}
        reviewed = checked["reviewed"]
        if reviewed is None:
            return {"status": "citation_validation_failed", "synthesis_calls": 1,
                    "synthesis": {"status": "citation_validation_failed", "answer": None,
                                  "claims": [], "validation_rejections": checked["rejections"],
                                  "human_review_required": True},
                    "trace": [{"node": "synthesize_answer", "event": "model_claims_rejected"}]}
        # Phase 4.3: recheck citation/source identity and attach an explicit
        # semantic-review requirement. Never claim automated entailment proof.
        try:
            faithfulness_audit = audit_checked_answer(reviewed, register)
        except (FaithfulnessError, ValueError, TypeError, KeyError):
            return {"status": "citation_validation_failed", "synthesis_calls": 1,
                    "synthesis": {"status": "citation_validation_failed", "answer": None,
                                  "claims": [], "human_review_required": True},
                    "trace": [{"node": "synthesize_answer", "event": "faithfulness_audit_rejected"}]}
        return {"status": reviewed["status"], "synthesis_calls": 1,
                "synthesis": {**reviewed, "source_register": public_register,
                              "faithfulness_audit": faithfulness_audit},
                "trace": [{"node": "synthesize_answer", "event": reviewed["status"]}]}

    def finish(state: ExecutionState) -> dict:
        return {"trace": [{"node": "finalize", "event": (
            "answer_review_required" if state.get("synthesis", {}).get("answer") else "no_answer_generated"
        )}]}

    graph = StateGraph(ExecutionState)
    graph.add_node("validate_request", validate_request_node)
    graph.add_node("plan_research", lambda state: plan_research_node(state, planner_provider, query_style=options.query_style))
    graph.add_node("select_source", choose_source)
    graph.add_node("discover", do_discovery)
    graph.add_node("prepare", prepare)
    graph.add_node("retrieve", retrieve_one)
    if assess_mode or synthesize_mode:
        graph.add_node("assess_evidence", assess_node)
        graph.add_node("retry_evidence", retry_one)
    if synthesize_mode:
        graph.add_node("synthesize_answer", synthesize_node)
    graph.add_node("finalize", finish)
    graph.add_edge(START, "validate_request")
    graph.add_conditional_edges("validate_request", lambda state: "finalize" if state.get("status") == "invalid_request" else "plan_research")
    graph.add_edge("plan_research", "select_source")
    graph.add_conditional_edges("select_source", lambda state: (
        "prepare" if state.get("status") == "source_selected" else
        "discover" if state.get("status") == "discover_requested" else "finalize"))
    graph.add_conditional_edges("discover", lambda state: "prepare" if state.get("status") == "source_selected" else "finalize")
    graph.add_conditional_edges("prepare", lambda state: "retrieve" if state.get("status") == "prepared_integrity_checked" else "finalize")
    graph.add_conditional_edges("retrieve", lambda state: (
        "retrieve" if state.get("status") == "retrieving" else
        "assess_evidence" if (assess_mode or synthesize_mode) and state.get("status") == "retrieval_complete" else "finalize"
    ))
    if assess_mode or synthesize_mode:
        graph.add_conditional_edges("assess_evidence", lambda state: (
            "retry_evidence" if should_retry(state) == "retry" else
            "synthesize_answer" if synthesize_mode else "finalize"
        ))
        graph.add_conditional_edges("retry_evidence", lambda state: (
            "assess_evidence" if state.get("status") == "retry_complete" else "finalize"
        ))
    if synthesize_mode:
        graph.add_edge("synthesize_answer", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()



def _validate_evidence_passages(passages: object, *, paper_id: str, top_k: int) -> list[dict]:
    """Reject malformed primary hits; never trust tool-provided page URLs/context."""
    if not isinstance(passages, list) or len(passages) > top_k:
        raise ValueError("Invalid retrieval results")
    approved: list[dict] = []
    for item in passages:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError("Invalid evidence passage")
        page = item["page_number"]
        if type(page) is not int or page < 1 or len(item["text"]) > 20_000:
            raise ValueError("Invalid page or text size")
        citation = f"https://arxiv.org/abs/{paper_id}#page={page}"
        if item.get("citation") != citation:
            raise ValueError("Evidence citation mismatch")
        chunk_id = item["chunk_id"]
        rank = item["rank"]
        rrf_score = item["rrf_score"]
        if not isinstance(chunk_id, str) or not chunk_id or len(chunk_id) > 300:
            raise ValueError("Invalid chunk ID")
        if type(rank) is not int or rank < 1 or rank > top_k:
            raise ValueError("Invalid retrieval rank")
        if isinstance(rrf_score, bool) or not isinstance(rrf_score, (float, int)) or not 0 <= rrf_score <= 100:
            raise ValueError("Invalid fusion score")
        context = item.get("context", [])
        if not isinstance(context, list) or len(context) > 4:
            raise ValueError("Invalid adjacent context")
        safe_context = []
        for c in context:
            if not isinstance(c, dict) or c.get("page_number") != page or c.get("citation") != citation:
                raise ValueError("Context citation mismatch")
            if not isinstance(c.get("text"), str) or len(c["text"]) > 20_000:
                raise ValueError("Invalid context text")
            if not isinstance(c.get("chunk_id"), str) or len(c["chunk_id"]) > 300:
                raise ValueError("Invalid context chunk id")
            safe_context.append({"chunk_id": c["chunk_id"], "page_number": page,
                                 "citation": citation, "text": c["text"]})
        approved.append({"chunk_id": chunk_id, "page_number": page,
                         "citation": citation, "rank": rank, "rrf_score": float(rrf_score),
                         "text": item["text"], "context": safe_context})
    if len({p["chunk_id"] for p in approved}) != len(approved):
        raise ValueError("Duplicate chunk IDs in one retrieval response")
    return approved


def _executed_plan(plan: list[dict], events: list[ToolEvent], evidence: list[dict],
                   assessment: dict | None = None, synthesis: dict | None = None) -> list[dict]:
    """Annotate the proposed DAG without implying assess/answer were run."""
    statuses: dict[str, str] = {}
    event_names = {e["name"]: e["status"] for e in events}
    if event_names.get("arxiv_search") == "success":
        statuses["discover"] = "completed"
    if event_names.get("pdf_ingestion") in {"success", "reused_existing_artifact", "skipped_using_downstream_chunks"}:
        statuses["ingest"] = (
            "completed" if event_names["pdf_ingestion"] == "success" else
            "cached" if event_names["pdf_ingestion"] == "reused_existing_artifact" else
            "skipped_missing_original"
        )
    if event_names.get("index_document") == "success":
        source_reused = all(event_names.get(name) == "reused_existing_artifact" for name in (
            "pdf_ingestion", "chunk_document", "embed_document"))
        index_cached = next((e["detail"].get("cached", False) for e in reversed(events)
                             if e["name"] == "index_document"), False)
        statuses["prepare"] = "cached" if source_reused and index_cached else "completed"
    for result in evidence:
        statuses[result["step_id"]] = "completed" if result["hits"] else "completed_no_hits"
    if assessment is not None:
        statuses["assess"] = "completed_structural_unverified"
    if synthesis is not None:
        statuses["answer"] = (
            "completed_cited_unverified" if synthesis.get("answer") else
            "preview_not_executed" if synthesis.get("status") == "ready_for_llm_synthesis" else
            "abstained_or_rejected"
        )
    return [{**task, "execution": statuses.get(task["id"], task["execution"])} for task in plan]

def run_agent_research(
    question: str, *, paper_id: str | None = None, options: ExecutionOptions | None = None,
    tools: ResearchTools | None = None, planner_mode: Literal["deterministic", "llm"] = "deterministic",
    planner_api_key: str | None = None, planner_model: str = "gpt-4.1-mini",
    planner_base_url: str = "https://api.openai.com/v1",
    planner_timeout: float = 20.0,
    assess_mode: bool = False,
    synthesize_mode: bool = False,
    synthesis_provider: Callable[[dict], dict] | None = None,
    on_progress: Callable[[int, str], None] | None = None,
) -> dict:
    """Execute real research tools; optional milestones reflect completed nodes only."""
    options = options or ExecutionOptions()
    request = ResearchRequest(question=question, max_papers=options.max_papers)
    if paper_id is not None:
        paper_id = validate_paper_id(paper_id)
    if planner_mode not in {"deterministic", "llm"}:
        raise ExecutionError("Unknown planner mode")
    if options.query_style in {"focused", "multi", "budgeted"} and (planner_mode != "deterministic" or synthesis_provider is not None or options.allow_network):
        raise ExecutionError("Experimental query styles are diagnostic only: offline deterministic preview-only; no LLM or network")
    if synthesize_mode and synthesis_provider is None and planner_mode == "llm":
        raise ExecutionError("Synthesis preview does not allow any LLM planner calls")
    provider = None
    if planner_mode == "llm" and planner_api_key:
        from .llm_planner import OpenAICompatibleQuestionPlanner
        provider = OpenAICompatibleQuestionPlanner(
            api_key=planner_api_key, model=planner_model,
            base_url=planner_base_url, timeout=planner_timeout,
        )
    graph = build_tool_graph(options, tools=tools, planner_provider=provider,
                             assess_mode=assess_mode, synthesize_mode=synthesize_mode,
                             synthesis_provider=synthesis_provider)
    initial_state = {
        "question": request.question, "paper_id": paper_id, "max_papers": options.max_papers,
        "planner_requested": planner_mode, "trace": [], "tool_calls": 0, "tool_events": [],
        "retrieval_cursor": 0, "evidence": [], "errors": [], "retry_attempts": {},
    }
    with _offline_model_guard(not options.allow_network):
        if on_progress is None:
            # Original invocation path is unchanged for CLI, benchmarks and tests.
            output = graph.invoke(initial_state, config={"recursion_limit": 48})
        else:
            # Stream state values on the calling thread. Streamlit callbacks must
            # never run in LangGraph worker threads; this also avoids fake timers.
            on_progress(0, 'Starting research workflow')
            output = None
            last_percent = 0
            for state in graph.stream(initial_state, config={"recursion_limit": 48}, stream_mode="values"):
                output = state
                checkpoint = research_checkpoint(state)
                if checkpoint is not None:
                    percent, label = checkpoint
                    # Loops/retries never make the progress bar move backwards.
                    last_percent = max(last_percent, percent)
                    on_progress(last_percent, label)
            if output is None:
                raise ExecutionError('Research graph produced no state updates')
    return {
        "milestone": "3.5" if synthesize_mode else "3.4" if assess_mode else "3.3",
        "status": output.get("status", "unknown"),
        "engine": "langgraph_stategraph", "paper_id": output.get("paper_id"),
        "question_analysis": output.get("question_analysis"),
        "plan_validation": output.get("plan_validation"),
        "query_style": options.query_style,
        "plan": _executed_plan(output.get("plan", []), output.get("tool_events", []),
                               output.get("evidence", []),
                               output.get("assessment") if (assess_mode or synthesize_mode) else None,
                               output.get("synthesis") if synthesize_mode else None),
        "planner": {"requested": planner_mode, "used": output.get("planner_used", "not_run"),
                    "fallback_reason": output.get("planner_fallback_reason")},
        "discovery_candidates": output.get("discovery", []),
        "discovery_selection_unverified": bool(paper_id is None and options.auto_select_first and output.get("paper_id")),
        "source_audit": output.get("source_audit"),
        "tool_calls": output.get("tool_calls", 0),
        "llm_calls": output.get("llm_calls", 0) + output.get("synthesis_calls", 0),
        "tool_events": output.get("tool_events", []),
        "retrieval_steps_completed": len(output.get("evidence", [])),
        "retrieval_steps_planned": sum(task.get("proposed_tool") == "hybrid_retrieval" for task in output.get("plan", [])),
        "primary_evidence_hits": sum(len(e["hits"]) for e in output.get("evidence", [])),
        "evidence": output.get("evidence", []), "trace": output.get("trace", []),
        "errors": output.get("errors", []),
        "network_opt_in": options.allow_network,
        "research_executed": bool(output.get("tool_calls", 0)),
        "answer_generated": bool(output.get("synthesis", {}).get("answer")) if synthesize_mode else False,
        **({"synthesis": output.get("synthesis", {}),
            "answer": output.get("synthesis", {}).get("answer"),
            "generation_mode": "live" if synthesis_provider is not None else "preview",
            "citation_quote_anchoring_verified": bool(output.get("synthesis", {}).get("citation_quote_anchoring_verified", False)),
            "semantic_entailment_verified": False,
            "human_review_required": True} if synthesize_mode else {}),
        **({"assessment": output.get("assessment"),
            "retry_attempts": output.get("retry_attempts", {}),
            "retry_justification": output.get("retry_justification", []),
            "review_requires_human_verification": True} if (assess_mode or synthesize_mode) else {}),
        "limitations": [
            ("Evidence is only structurally assessed, not semantically or factually verified."
             if assess_mode else "Evidence is retrieved, not yet assessed for sufficiency, correctness, or contradictions."),
            "PDF extraction order and source-page relevance still need independent human audit.",
            ("Generated claims are quote-anchored but semantically UNVERIFIED; no independent entailment or contradiction checking."
             if synthesize_mode else "No answer synthesis, autonomous iterative correction, or persistent checkpoint yet."),
            "Source content is untrusted data; citations identify source pages, not verified claims.",
        ],
    }

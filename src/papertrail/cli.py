"""CLI for local evidence-first retrieval, analysis and evaluation."""

import argparse
import json
import sys
from pathlib import Path
from collections.abc import Sequence

from pydantic import ValidationError
from pydantic_settings import SettingsError

from .chunking import ChunkingError, chunk_saved_document
from .chunking.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from .config import Settings
from .discovery import ArxivClient, ArxivParseError, ArxivRequestError
from .embedding import EmbeddingError, embed_saved_document
from .health import doctor_summary
from .ingestion import ArxivPdfDownloader, PdfDownloadError, PdfExtractionError, ingest_arxiv_pdf
from .models import ResearchRequest
from .retrieval import RetrievalError, build_paper_index, search_paper
from .retrieval.hybrid import hybrid_search_paper
from .retrieval.evaluation import compare_benchmark_retrieval
from .retrieval.quality_eval import run_retrieval_evaluation, save_retrieval_report
from .retrieval.optimization import run_retrieval_optimization, save_optimization_report
from .agent import AgentDependencyError, run_agent_preview
from .agent.planning import PlanningError
from .agent.tool_execution import ExecutionError, ExecutionOptions, run_agent_research
from .agent.planning_eval import evaluate_planning_suite, save_planning_report
from .agent.synthesis import SynthesisError, OpenAICompatibleAnswerGenerator
from .agent.evaluation import (AgentEvaluationError, load_agent_suite, parse_k_values,
                               run_agent_evaluation, save_agent_evaluation)
from .agent.query_eval import run_query_evaluation, save_query_evaluation
from .agent.multi_eval import run_multi_query_evaluation, save_multi_query_evaluation
from .agent.budget_eval import run_budget_evaluation, save_budget_evaluation
from .agent.faithfulness import (FaithfulnessError, _read_json, audit_bundle, load_faithfulness_suite,
                                 run_faithfulness_suite, save_audit_queue, save_faithfulness_suite_report, summarize_manual_review)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="papertrail", description="PaperTrail research metadata discovery")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect local configuration without exposing secrets")
    validate = commands.add_parser("validate-question", help="Validate a research question without API calls")
    validate.add_argument("--question", required=True)
    search = commands.add_parser("search-papers", help="Search public arXiv metadata (requires internet)")
    search.add_argument("--query", required=True, help='Keywords, e.g. "retrieval augmented generation"')
    search.add_argument("--limit", type=int, default=None, help="Number of papers (1–20); defaults to project setting")
    search.add_argument("--start", type=int, default=0, help="Start at this result offset (0 based)")
    search.add_argument("--sort-by", choices=["relevance", "submittedDate", "lastUpdatedDate"], default="relevance")
    search.add_argument("--sort-order", choices=["ascending", "descending"], default="descending")
    ingest = commands.add_parser("ingest-pdf", help="Download and extract a single arXiv PDF")
    ingest.add_argument("--paper-id", required=True, help="arXiv identifier, e.g. 2510.22344")
    ingest.add_argument("--max-pages", type=int, default=None, help="Maximum pages to extract (1–200)")
    ingest.add_argument("--force", action="store_true", help="Re-download instead of using existing local PDF")
    ingest.add_argument("--layout", choices=["auto", "simple"], default="auto",
                        help="Auto-detect two-column layout or use legacy extraction")
    chunk = commands.add_parser("chunk-document", help="Split a locally extracted PDF JSON into page-aware chunks (offline)")
    chunk.add_argument("--paper-id", required=True, help="arXiv identifier, e.g. 2510.22344")
    chunk.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Maximum characters per chunk (100–8000)")
    chunk.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Overlapping characters between chunks (less than half chunk-size)")
    embed = commands.add_parser("embed-document", help="Encode local chunks with pinned all-MiniLM-L6-v2")
    embed.add_argument("--paper-id", required=True, help="arXiv identifier")
    embed.add_argument("--batch-size", type=int, default=16, help="Texts per forward batch (1–128)")
    embed.add_argument("--force", action="store_true", help="Recompute despite matching cached embeddings")
    index = commands.add_parser("index-document", help="Build or verify a local FAISS cosine index (offline)")
    index.add_argument("--paper-id", required=True, help="arXiv identifier")
    index.add_argument("--force", action="store_true", help="Rebuild even when the saved index is valid")
    query = commands.add_parser("search-chunks", help="Semantic search with source pages (local embeddings; no API key)")
    query.add_argument("--paper-id", required=True, help="arXiv identifier")
    query.add_argument("--query", required=True, help="A research question or search phrase")
    query.add_argument("--top-k", type=int, default=5, help="Number of cited chunks to return (1-20)")
    hybrid = commands.add_parser("hybrid-search", help="Fuse FAISS and BM25 with RRF; return source-cited neighbor context")
    hybrid.add_argument("--paper-id", required=True, help="arXiv identifier")
    hybrid.add_argument("--query", required=True, help="Research question")
    hybrid.add_argument("--top-k", type=int, default=5, help="Number of primary hits (1-20)")
    hybrid.add_argument("--context-window", type=int, default=1, help="Number of adjacent chunks per side (0-2, same page only)")
    hybrid.add_argument("--dense-weight", type=float, default=1.0, help="RRF FAISS weight (0-10; default 1)")
    hybrid.add_argument("--bm25-weight", type=float, default=1.0, help="RRF BM25 weight (0-10; default 1)")
    hybrid.add_argument("--max-hits-per-page", type=int, default=0, help="Optional page diversity: 0 off, or 1-5")
    evaluate = commands.add_parser("evaluate-benchmarks", help="Compare literal benchmark-name coverage: FAISS vs hybrid")
    evaluate.add_argument("--paper-id", required=True, help="arXiv identifier")
    evaluate.add_argument("--top-k", type=int, default=10, help="Number of primary hits (1-20)")
    evaluate.add_argument("--context-window", type=int, default=1, help="Hybrid context radius (0-2)")
    eval_retrieval = commands.add_parser("evaluate-retrieval", help="Evaluate BM25, FAISS and hybrid using labeled PDF pages")
    eval_retrieval.add_argument("--paper-id", required=True, help="arXiv identifier")
    eval_retrieval.add_argument("--suite", default="benchmarks/fair_rag_pages_v1.json", help="Local page-relevance JSON suite")
    eval_retrieval.add_argument("--k-values", default="1,3,5,10", help="Comma-separated increasing K values, at most 20")
    eval_retrieval.add_argument("--dense-weight", type=float, default=1.0, help="RRF FAISS weight (0-10)")
    eval_retrieval.add_argument("--bm25-weight", type=float, default=1.0, help="RRF BM25 weight (0-10)")
    eval_retrieval.add_argument("--max-hits-per-page", type=int, default=0, help="Optional page diversity: 0 off, or 1-5")
    optimize = commands.add_parser("optimize-retrieval", help="Audit provisional qrels and compare frozen weighted RRF grid with train/holdout split")
    optimize.add_argument("--paper-id", required=True)
    optimize.add_argument("--suite", default="benchmarks/fair_rag_pages_v1.json")
    optimize.add_argument("--k", type=int, default=5)
    optimize.add_argument("--metric", choices=["mrr_at_k", "recall_at_k", "precision_at_k", "hit_at_k"], default="mrr_at_k")
    optimize.add_argument("--seed", type=int, default=42)
    preview = commands.add_parser("agent-preview", help="Run an auditable research planner; LLM questions opt-in, no research executed")
    preview.add_argument("--question", required=True, help="Research question (12-1000 characters)")
    preview.add_argument("--paper-id", default=None, help="Optional arXiv ID for local file presence preflight")
    preview.add_argument("--planner", choices=["deterministic", "llm"], default="deterministic",
                         help="Question-aware offline rules (default), or opt-in LLM subquestion suggestions")
    research = commands.add_parser("agent-research", help="Execute bounded real research tools; retrieve page-cited evidence, no answer generation")
    research.add_argument("--question", required=True)
    research.add_argument("--paper-id", default=None, help="Existing arXiv ID; omit to opt into discovery with --allow-network")
    research.add_argument("--planner", choices=["deterministic", "llm"], default="deterministic")
    research.add_argument("--top-k", type=int, default=3, help="Maximum source hits per subquestion (1-10)")
    research.add_argument("--context-window", type=int, default=1, help="Same-page context radius (0-2)")
    research.add_argument("--max-tool-calls", type=int, default=12, help="Hard cap on actual tool invocations (1-20)")
    research.add_argument("--allow-network", action="store_true", help="Explicit opt-in for public arXiv search/PDF downloads")
    research.add_argument("--auto-select-first", action="store_true", help="Explicit opt-in: use the first arXiv discovery result; review its relevance first")
    assess = commands.add_parser("agent-assess", help="Run bounded research tools, structurally assess evidence gaps, and optionally retry; NO answer")
    assess.add_argument("--question", required=True)
    assess.add_argument("--paper-id", default=None)
    assess.add_argument("--planner", choices=["deterministic", "llm"], default="deterministic")
    assess.add_argument("--top-k", type=int, default=3)
    assess.add_argument("--context-window", type=int, default=1)
    assess.add_argument("--max-tool-calls", type=int, default=12)
    assess.add_argument("--min-hits-per-step", type=int, default=2,
                        help="Minimum DISTINCT primary chunk IDs per facet (1-10; structural proxy only)")
    assess.add_argument("--max-refinement-rounds", type=int, default=1,
                        help="0 disables retries; 1 uses a safer user-question fallback; maximum 1 (no repeated identical queries)")
    assess.add_argument("--allow-network", action="store_true", help="Opt-in to arXiv and model downloads only")
    assess.add_argument("--auto-select-first", action="store_true", help="Explicitly select unverified first arXiv result")
    answer = commands.add_parser("agent-answer", help="Preview evidence or explicitly generate quote-anchored cited claims with an LLM")
    answer.add_argument("--question", required=True)
    answer.add_argument("--paper-id", default=None, help="Existing arXiv ID (recommended)")
    answer.add_argument("--mode", choices=["preview", "live"], default="preview",
                        help="Preview: NO LLM/key. Live: one paid-capable LLM request, needs OPENAI_API_KEY")
    answer.add_argument("--planner", choices=["deterministic", "llm"], default="deterministic")
    answer.add_argument("--query-style", choices=["baseline", "focused", "multi", "budgeted"], default="baseline",
                        help="Focused/Multi/Budgeted: exploratory OFFLINE deterministic preview only")
    answer.add_argument("--top-k", type=int, default=3)
    answer.add_argument("--context-window", type=int, default=1)
    answer.add_argument("--max-tool-calls", type=int, default=12)
    answer.add_argument("--min-hits-per-step", type=int, default=2)
    answer.add_argument("--max-refinement-rounds", type=int, default=1)
    answer.add_argument("--allow-network", action="store_true", help="Opt into public arXiv/PDF/model-weight downloads")
    answer.add_argument("--auto-select-first", action="store_true", help="Opt into selecting unreviewed first arXiv result")
    agent_evaluate = commands.add_parser("agent-evaluate", help="Offline Phase 4.1 retrieval/selection diagnostics using PROVISIONAL log-derived anchors; NO API")
    agent_evaluate.add_argument("--suite", default="benchmarks/agent_fairrag_log_anchors_v1.json",
                                help="Local, provisional log-derived reference suite")
    agent_evaluate.add_argument("--k-values", default="3,5", help="Increasing unique top-k values, up to four, between 1 and 10")
    agent_evaluate.add_argument("--recorded-live", default=None, help="Optional sanitized historical live summary JSON, NEVER a new LLM call")
    query_eval = commands.add_parser("agent-query-evaluate", help="Phase 4.2 OFFLINE baseline/focused A/B on provisional anchors; NO API")
    query_eval.add_argument("--suite", default="benchmarks/agent_fairrag_log_anchors_v1.json",
                            help="Existing provisional case/anchor suite")
    query_eval.add_argument("--k-values", default="3,5", help="Increasing unique k values (1..10), up to four")
    multi_eval = commands.add_parser("agent-multi-query-evaluate", help="Phase 4.2.1 OFFLINE baseline/focused/multi fixed-budget ablation; NO API")
    multi_eval.add_argument("--suite", default="benchmarks/agent_fairrag_log_anchors_v1.json",
                            help="Existing provisional case/anchor suite")
    multi_eval.add_argument("--k-values", default="3,5", help="Increasing unique k values (1..10), up to four")
    budget_eval = commands.add_parser("agent-budget-evaluate", help="Phase 4.2.2 OFFLINE baseline/focused/RRF/budgeted fusion ablation; NO API")
    budget_eval.add_argument("--suite", default="benchmarks/agent_fairrag_log_anchors_v1.json",
                             help="Provisional diagnostic anchor suite (not independently labeled)")
    budget_eval.add_argument("--k-values", default="3,5", help="Increasing unique k values (1..10), up to four")
    faith_eval = commands.add_parser("agent-faithfulness-evaluate", help="Phase 4.3 offline synthetic quote-guard regression; NOT semantic accuracy")
    faith_eval.add_argument("--suite", default="benchmarks/faithfulness_synthetic_v1.json")
    faith_audit = commands.add_parser("agent-faithfulness-audit", help="Audit locally supplied claim/evidence bundle and export private manual-review queue")
    faith_audit.add_argument("--input", required=True, help="Local JSON with plan, register and draft; no API")
    faith_audit.add_argument("--name", default="manual_review_v1", help="Safe unique filename stem, existing file is NOT overwritten")
    faith_sum = commands.add_parser("agent-faithfulness-review-summary", help="Summarize unverified user-entered semantic review decisions; no API")
    faith_sum.add_argument("--review-file", required=True)
    planning_eval = commands.add_parser("evaluate-planning", help="Evaluate deterministic planner on synthetic labeled regression questions")
    planning_eval.add_argument("--suite", default="benchmarks/planning_cases_v1.json", help="Local synthetic planning benchmark JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = Settings()
        if args.command == "agent-faithfulness-evaluate":
            suite = load_faithfulness_suite(Path(args.suite))
            report = run_faithfulness_suite(suite)
            json_path, md_path = save_faithfulness_suite_report(report, settings.data_dir)
            print(json.dumps({**{k:v for k,v in report.items() if k != 'per_case'},
                              "report_json": str(json_path), "report_markdown": str(md_path)},
                             ensure_ascii=False, indent=2))
            return 0 if report["checks_failed"] == 0 else 1
        if args.command == "agent-faithfulness-review-summary":
            print(json.dumps(summarize_manual_review(Path(args.review_file)),
                             ensure_ascii=False, indent=2))
            return 0
        if args.command == "agent-faithfulness-audit":
            report = audit_bundle(_read_json(Path(args.input)))
            path = save_audit_queue(report, settings.data_dir, args.name)
            print(json.dumps({"phase": "4.3", "status": report["status"],
                              "citation_validation_status": report["citation_validation_status"],
                              "accepted_claims": report['audit']['accepted_claim_count'],
                              "rejected_claims": len(report['rejections']),
                              "verbatim_within_quote": report['audit']['verbatim_within_quote_count'],
                              "nonverbatim_needing_review": report['audit']['quote_anchored_nonverbatim_count'],
                              "human_review_required": True,
                              "semantic_entailment_verified": False,
                              "independent_pdf_provenance_verified": False,
                              "llm_calls": 0, "network_opt_in": False,
                              "review_queue_file": str(path)}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "agent-budget-evaluate":
            suite = load_agent_suite(Path(args.suite))
            report = run_budget_evaluation(suite, data_dir=settings.data_dir,
                                           k_values=parse_k_values(args.k_values))
            json_path, md_path = save_budget_evaluation(report, settings.data_dir)
            print(json.dumps({
                "phase": "4.2.2", "status": report["status"], "suite_id": report["suite_id"],
                "annotation_status": report["annotation_status"],
                "independently_labeled_ground_truth": False,
                "semantic_entailment_verified": False,
                "default_agent_query_style_changed": False,
                "llm_calls": 0, "network_opt_in": False,
                "k_values": report["k_values"], "case_count": report["case_count"],
                "run_count": report["run_count"], "comparisons": report["comparisons"],
                "interpretation": report["interpretation"],
                "report_json": str(json_path), "report_markdown": str(md_path),
            }, ensure_ascii=False, indent=2))
            return 0
        if args.command == "agent-multi-query-evaluate":
            suite = load_agent_suite(Path(args.suite))
            report = run_multi_query_evaluation(suite, data_dir=settings.data_dir,
                                                k_values=parse_k_values(args.k_values))
            json_path, md_path = save_multi_query_evaluation(report, settings.data_dir)
            print(json.dumps({
                "phase": "4.2.1", "status": report["status"], "suite_id": report["suite_id"],
                "annotation_status": report["annotation_status"],
                "independently_labeled_ground_truth": False,
                "semantic_entailment_verified": False,
                "default_agent_query_style_changed": False,
                "llm_calls": 0, "network_opt_in": False,
                "k_values": report["k_values"], "case_count": report["case_count"],
                "run_count": report["run_count"], "comparisons": report["comparisons"],
                "interpretation": report["interpretation"],
                "report_json": str(json_path), "report_markdown": str(md_path),
            }, ensure_ascii=False, indent=2))
            return 0
        if args.command == "agent-query-evaluate":
            suite = load_agent_suite(Path(args.suite))
            report = run_query_evaluation(suite, data_dir=settings.data_dir,
                                          k_values=parse_k_values(args.k_values))
            json_path, md_path = save_query_evaluation(report, settings.data_dir)
            print(json.dumps({
                "phase": "4.2", "status": report["status"], "suite_id": report["suite_id"],
                "annotation_status": report["annotation_status"],
                "independently_labeled_ground_truth": False,
                "semantic_entailment_verified": False,
                "default_agent_query_style_changed": False,
                "llm_calls": 0, "network_opt_in": False,
                "k_values": report["k_values"], "case_count": report["case_count"],
                "run_count": report["run_count"], "comparisons": report["comparisons"],
                "interpretation": report["interpretation"],
                "report_json": str(json_path), "report_markdown": str(md_path),
            }, ensure_ascii=False, indent=2))
            return 0
        if args.command == "agent-evaluate":
            suite = load_agent_suite(Path(args.suite))
            k_values = parse_k_values(args.k_values)
            report = run_agent_evaluation(
                suite, data_dir=settings.data_dir, k_values=k_values,
                recorded_live=Path(args.recorded_live) if args.recorded_live else None,
            )
            json_path, md_path = save_agent_evaluation(report, settings.data_dir)
            print(json.dumps({
                "phase": "4.1", "status": report["status"], "suite_id": report["suite_id"],
                "annotation_status": report["annotation_status"], "independently_labeled_ground_truth": False,
                "semantic_entailment_verified": False, "llm_calls": 0, "network_opt_in": False,
                "configs": report["configs"], "case_count": report["case_count"],
                "run_count": report["run_count"], "per_run": report["per_run"],
                "recorded_live_observation": report["recorded_live_observation"],
                "interpretation": report["interpretation"], "report_json": str(json_path),
                "report_markdown": str(md_path),
            }, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"agent-research", "agent-assess"}:
            assessment_mode = args.command == "agent-assess"
            options = ExecutionOptions(
                data_dir=settings.data_dir, top_k=args.top_k, context_window=args.context_window,
                max_tool_calls=args.max_tool_calls, allow_network=args.allow_network,
                auto_select_first=args.auto_select_first, max_papers=settings.max_papers,
                max_pdf_pages=settings.max_pdf_pages, max_retries=settings.max_retries,
                timeout_seconds=settings.request_timeout_seconds,
                **({"min_hits_per_step": args.min_hits_per_step,
                    "max_refinement_rounds": args.max_refinement_rounds} if assessment_mode else {}),
            )
            result = run_agent_research(
                args.question, paper_id=args.paper_id, options=options,
                planner_mode=args.planner,
                planner_api_key=(settings.openai_api_key.get_secret_value() if settings.openai_api_key else None)
                    if args.planner == "llm" else None,
                planner_model=settings.model_name, planner_base_url=settings.openai_base_url,
                planner_timeout=settings.request_timeout_seconds,
                **({"assess_mode": True} if assessment_mode else {}),
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] in {
                "evidence_retrieved_no_answer", "paper_selection_required", "needs_paper_selection",
                "no_papers_found", "no_retrieval_hits", "evidence_assessed_unverified",
                "evidence_gaps_detected", "no_retrieval_steps",
            } else 1
        if args.command == "agent-answer":
            if args.query_style in {"focused", "multi", "budgeted"} and (args.mode != "preview" or args.planner != "deterministic" or args.allow_network):
                raise SynthesisError("Experimental query styles are OFFLINE deterministic preview-only; not enabled for live, network or LLM planning")
            if args.mode == "preview" and args.planner == "llm":
                raise SynthesisError("Preview always uses --planner deterministic; an LLM planner could use an API key")
            # Live-mode credential check happens BEFORE retrieval or any network work.
            generator = None
            if args.mode == "live":
                if not settings.openai_api_key or not settings.openai_api_key.get_secret_value():
                    raise SynthesisError("Live synthesis needs OPENAI_API_KEY in project-root .env; never paste keys into chat or GitHub")
                generator = OpenAICompatibleAnswerGenerator(
                    api_key=settings.openai_api_key.get_secret_value(),
                    model=settings.model_name, base_url=settings.openai_base_url,
                    timeout=settings.request_timeout_seconds,
                )
            options = ExecutionOptions(
                data_dir=settings.data_dir, top_k=args.top_k, context_window=args.context_window,
                max_tool_calls=args.max_tool_calls, allow_network=args.allow_network,
                auto_select_first=args.auto_select_first, max_papers=settings.max_papers,
                max_pdf_pages=settings.max_pdf_pages, max_retries=settings.max_retries,
                timeout_seconds=settings.request_timeout_seconds,
                min_hits_per_step=args.min_hits_per_step,
                max_refinement_rounds=args.max_refinement_rounds,
                query_style=args.query_style,
            )
            result = run_agent_research(
                args.question, paper_id=args.paper_id, options=options,
                planner_mode=args.planner,
                planner_api_key=(settings.openai_api_key.get_secret_value()
                                 if args.planner == "llm" and settings.openai_api_key else None),
                planner_model=settings.model_name, planner_base_url=settings.openai_base_url,
                planner_timeout=settings.request_timeout_seconds,
                synthesize_mode=True, synthesis_provider=generator,
            )
            # Hide giant underlying source excerpts by default; the project still
            # returns a bounded source register and auditable claims.
            result.pop("evidence", None)
            result["evidence_content_suppressed_in_cli"] = True
            print(json.dumps(result, indent=2, ensure_ascii=False))
            ok = {"ready_for_llm_synthesis", "cited_answer_unverified", "partial_answer_unverified",
                  "insufficient_evidence", "needs_paper_selection", "paper_selection_required", "no_papers_found"}
            return 0 if result["status"] in ok else 1
        if args.command == "agent-preview":
            kwargs = {
                "paper_id": args.paper_id,
                "data_dir": settings.data_dir,
                "max_papers": settings.max_papers,
            }
            if args.planner == "llm":
                kwargs.update({
                    "planner_mode": "llm",
                    "planner_api_key": (
                        settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
                    ),
                    "planner_model": settings.model_name,
                    "planner_base_url": settings.openai_base_url,
                    "planner_timeout": settings.request_timeout_seconds,
                })
            result = run_agent_preview(args.question, **kwargs)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command == "evaluate-planning":
            report = evaluate_planning_suite(Path(args.suite))
            json_path, md_path = save_planning_report(report, settings.data_dir)
            summary = {k: v for k, v in report.items() if k != "per_case"}
            summary.update({"status": "synthetic_regression_evaluated", "report_json": str(json_path), "report_markdown": str(md_path)})
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        if args.command == "doctor":
            print(json.dumps(doctor_summary(settings), indent=2))
            return 0
        if args.command == "validate-question":
            request = ResearchRequest(question=args.question, max_papers=settings.max_papers)
            print(json.dumps({"status": "valid", "request": request.model_dump(), "research_executed": False}, indent=2))
            return 0
        if args.command == "index-document":
            built = build_paper_index(args.paper_id, data_dir=settings.data_dir, force=args.force)
            print(json.dumps({
                "status": "cached" if built.cached else "indexed",
                "paper_id": built.manifest.paper_id,
                "vector_count": built.manifest.vector_count,
                "embedding_dimension": built.manifest.dimension,
                "index_type": built.manifest.index_type,
                "metric": built.manifest.metric,
                "index_path": str(built.index_path),
                "manifest_path": str(built.manifest_path),
            }, indent=2, ensure_ascii=False))
            return 0
        if args.command == "search-chunks":
            result = search_paper(args.paper_id, args.query, data_dir=settings.data_dir, top_k=args.top_k)
            print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
            return 0
        if args.command == "hybrid-search":
            result = hybrid_search_paper(
                args.paper_id, args.query, data_dir=settings.data_dir,
                top_k=args.top_k, context_window=args.context_window,
                dense_weight=args.dense_weight, bm25_weight=args.bm25_weight,
                max_hits_per_page=args.max_hits_per_page,
            )
            print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
            return 0
        if args.command == "optimize-retrieval":
            report = run_retrieval_optimization(
                args.paper_id, data_dir=settings.data_dir, suite_path=Path(args.suite),
                seed=args.seed, metric=args.metric, k=args.k,
            )
            jpath, mpath = save_optimization_report(report, data_dir=settings.data_dir)
            print(json.dumps({
                "status": "evaluated_not_deployed", "paper_id": report.paper_id,
                "suite_id": report.suite_id, "annotation_status": report.annotation_status,
                "metric": report.metric, "k": report.k,
                "train_query_ids": report.train_query_ids,
                "validation_query_ids": report.validation_query_ids,
                "selected_dense_weight": report.selected_dense_weight,
                "selected_bm25_weight": report.selected_bm25_weight,
                "selected_page_cap": report.selected_page_cap,
                "baseline_validation_score": report.baseline_validation_score,
                "selected_validation_score": report.selected_validation_score,
                "bm25_validation_score": report.bm25_validation_score,
                "faiss_validation_score": report.faiss_validation_score,
                "deployment_recommendation": report.deployment_recommendation,
                "manual_review_pages": len(report.flagged_unlabeled_pages),
                "report_json": str(jpath), "report_markdown": str(mpath),
                "pdf_extraction_quality_verified": False,
            }, ensure_ascii=False, indent=2))
            return 0
        if args.command == "evaluate-retrieval":
            report = run_retrieval_evaluation(
                args.paper_id, data_dir=settings.data_dir,
                suite_path=Path(args.suite), k_values=args.k_values,
                dense_weight=args.dense_weight, bm25_weight=args.bm25_weight,
                max_hits_per_page=args.max_hits_per_page,
            )
            json_path, md_path = save_retrieval_report(report, data_dir=settings.data_dir)
            summary = {
                "status": "evaluated",
                "paper_id": report.paper_id,
                "suite_id": report.suite_id,
                "annotation_status": report.annotation_status,
                "questions": report.query_count,
                "k_values": report.k_values,
                "macro_results": {
                    method: {k: values.model_dump(mode="json") for k, values in metrics.items()}
                    for method, metrics in report.macro_results.items()
                },
                "report_json": str(json_path),
                "report_markdown": str(md_path),
                "methodology_note": report.methodology_note,
                "hybrid_dense_weight": report.dense_weight,
                "hybrid_bm25_weight": report.bm25_weight,
                "hybrid_page_cap": report.max_hits_per_page,
                "pdf_extraction_quality_verified": report.pdf_extraction_quality_verified,
            }
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        if args.command == "evaluate-benchmarks":
            result = compare_benchmark_retrieval(
                args.paper_id, data_dir=settings.data_dir,
                top_k=args.top_k, context_window=args.context_window,
            )
            print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
            return 0
        if args.command == "embed-document":
            run = embed_saved_document(args.paper_id, data_dir=settings.data_dir, batch_size=args.batch_size, force=args.force)
            result = run.manifest
            summary = {
                "status": "cached" if run.cached else "embedded",
                "paper_id": result.paper_id,
                "total_vectors": result.vector_count,
                "embedding_dimension": result.embedding_dimension,
                "model": result.model_name,
                "model_revision": result.model_revision,
                "max_sequence_length": result.max_sequence_length,
                "max_input_tokens": result.max_input_tokens,
                "normalized": result.normalized,
                "vectors_path": str(run.vectors_path),
                "manifest_path": str(run.manifest_path),
            }
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        if args.command == "chunk-document":
            result, output_path = chunk_saved_document(
                args.paper_id, data_dir=settings.data_dir, chunk_size=args.chunk_size, overlap=args.overlap
            )
            summary = {
                "status": "chunked",
                "paper_id": result.paper_id,
                "source_url": result.source_url,
                "total_pages": result.total_pages,
                "extracted_pages": result.extracted_pages,
                "empty_text_pages": result.empty_text_pages,
                "total_chunks": result.total_chunks,
                "chunk_size": result.chunk_size,
                "overlap": result.overlap,
                "truncated": result.truncated,
                "chunks_json_path": str(output_path),
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if args.command == "ingest-pdf":
            with ArxivPdfDownloader(timeout_seconds=settings.request_timeout_seconds, max_retries=settings.max_retries) as client:
                document, pdf_path, json_path, cached = ingest_arxiv_pdf(
                    args.paper_id,
                    data_dir=settings.data_dir,
                    client=client,
                    max_pages=settings.max_pdf_pages if args.max_pages is None else args.max_pages,
                    force=args.force,
                    layout=args.layout,
                )
            summary = {
                "status": "extracted",
                "paper_id": document.paper_id,
                "source_url": document.source_url,
                "total_pages": document.total_pages,
                "extracted_pages": document.extracted_pages,
                "truncated": document.truncated,
                "pages_without_text": document.empty_text_pages,
                "two_column_pages": [p.page_number for p in document.pages if p.extraction_method == "two_column"],
                "layout": args.layout,
                "characters_extracted": sum(page.character_count for page in document.pages),
                "cached_pdf": cached,
                "pdf_path": str(pdf_path),
                "extracted_json_path": str(json_path),
                "sha256": document.sha256,
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if args.command == "search-papers":
            with ArxivClient(timeout_seconds=settings.request_timeout_seconds, max_retries=settings.max_retries) as client:
                result = client.search(
                    args.query, max_results=settings.max_papers if args.limit is None else args.limit, start=args.start,
                    sort_by=args.sort_by, sort_order=args.sort_order,
                )
            print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
            return 0
    except (ValidationError, SettingsError, ChunkingError, EmbeddingError, RetrievalError, AgentDependencyError, PlanningError, ExecutionError, SynthesisError, AgentEvaluationError, FaithfulnessError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            reason = exc.errors()[0]["msg"]
        elif isinstance(exc, SettingsError):
            reason = "Invalid environment configuration"
        else:
            reason = str(exc)
        print(f"Validation error: {reason}", file=sys.stderr)
        return 2
    except (PdfDownloadError, PdfExtractionError) as exc:
        print(f"PDF ingestion error: {exc}", file=sys.stderr)
        return 1
    except (ArxivRequestError, ArxivParseError) as exc:
        print(f"arXiv error: {exc}", file=sys.stderr)
        return 1
    parser.error("Unknown command")
    return 2

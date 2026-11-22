"""Auditable LangGraph planning workflow (Phase 3.2).

Phase 3.2 adds question-aware dependency DAGs and optional bounded LLM
subquestion proposals, with a deterministic fallback. The workflow still
performs only planning and a read-only local artifact existence preflight. In particular:
 - no LLM calls, internet requests, PDF reads, model inference or retrieval
 - no claim that the source is trustworthy or its files are internally valid
 - no persistence or checkpointing (each graph.invoke call is independent)

Pure node functions are testable without LangGraph. The real StateGraph
integration is loaded lazily and covered by an optional native smoke test.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Callable, Literal, TypedDict

from pydantic import ValidationError

from papertrail.ingestion.pdf_downloader import validate_paper_id
from papertrail.models import ResearchRequest
from .planning import PlanningError, build_research_plan, analyze_question, validate_subquestions


class AgentDependencyError(RuntimeError):
    """The optional LangGraph backend isn't installed."""


class TraceEvent(TypedDict):
    node: str
    event: str


class PlanStep(TypedDict):
    id: str
    task: str
    execution: str


class ArtifactCheck(TypedDict):
    check_type: str
    ready: bool
    present: list[str]
    missing: list[str]


class AgentState(TypedDict, total=False):
    question: str
    paper_id: str | None
    max_papers: int
    data_dir: str
    status: str
    plan: list[PlanStep]
    question_analysis: dict
    planner_requested: str
    planner_used: str
    planner_fallback_reason: str | None
    llm_calls: int
    plan_validation: str
    artifact_check: ArtifactCheck
    errors: list[str]
    trace: Annotated[list[TraceEvent], operator.add]


ARTIFACT_LABELS = ("chunks", "embedding_vectors", "embedding_manifest", "faiss_index", "index_manifest")


def validate_request_node(state: AgentState) -> dict:
    try:
        request = ResearchRequest(
            question=state.get("question", ""), max_papers=state.get("max_papers", 5)
        )
        paper_id = state.get("paper_id")
        if paper_id is not None:
            paper_id = validate_paper_id(paper_id)
    except (ValidationError, ValueError, TypeError) as exc:
        if isinstance(exc, ValidationError):
            reason = str(exc.errors()[0].get("msg", "Invalid research request"))
        else:
            reason = str(exc)
        return {
            "status": "invalid_request", "errors": [reason],
            "trace": [{"node": "validate_request", "event": "rejected"}],
        }
    return {
        "question": request.question, "paper_id": paper_id,
        "max_papers": request.max_papers, "status": "validated",
        "trace": [{"node": "validate_request", "event": "accepted"}],
    }


def route_after_validation(state: AgentState) -> Literal["plan", "finalize"]:
    return "finalize" if state.get("status") == "invalid_request" else "plan"


def plan_research_node(
    state: AgentState, planner_provider: Callable[[str, list[str]], list[str]] | None = None,
    *, query_style: Literal["baseline", "focused", "multi", "budgeted"] = "baseline",
) -> dict:
    """Plan-only node. Optional LLM is allowed to suggest questions, never facts."""
    if query_style not in {"baseline", "focused", "multi", "budgeted"}:
        raise PlanningError("Unknown retrieval query style")
    question = state["question"]
    paper_id = state.get("paper_id")
    mode = state.get("planner_requested", "deterministic")
    if mode not in {"deterministic", "llm"}:
        raise PlanningError("Unknown planner mode")
    preliminary = analyze_question(question, paper_id=paper_id)
    suggested = None
    used = "deterministic"
    calls = 0
    fallback_reason = None
    if mode == "llm":
        if planner_provider is None:
            fallback_reason = "provider_not_configured"
        else:
            calls = 1
            try:
                suggested = validate_subquestions(
                    planner_provider(question, preliminary.subquestions), question
                )
                used = "llm_proposal_unverified"
            except Exception:
                # A provider is untrusted and may return malformed content or fail.
                # Failure details must not leak provider headers or secret values.
                fallback_reason = "llm_provider_failed_or_output_rejected"
    plan = build_research_plan(question, paper_id=paper_id, proposed_subquestions=suggested)
    if query_style == "focused":
        if mode != "deterministic":
            raise PlanningError("Focused query style requires deterministic planner")
        from .query_rewriting import focused_queries
        alternatives = focused_queries(question, plan.analysis.intents)
        iterator = iter(alternatives)
        plan = plan.model_copy(update={
            "steps": [step.model_copy(update={"search_query": next(iterator)})
                      if step.proposed_tool == "hybrid_retrieval" else step
                      for step in plan.steps],
        })
        # Preserve human-readable subquestions. Only retrieval search terms change.
        plan = type(plan).model_validate(plan.model_dump())
    return {
        "plan": [step.model_dump(mode="json") for step in plan.steps],
        "question_analysis": plan.analysis.model_dump(mode="json"),
        "plan_validation": plan.validation,
        "query_style": query_style,
        "planner_requested": mode,
        "planner_used": used,
        "planner_fallback_reason": fallback_reason,
        "llm_calls": calls,
        "trace": [{"node": "plan_research", "event": (
            "llm_questions_proposed_unverified" if used != "deterministic" else
            "deterministic_fallback" if mode == "llm" else "question_aware_plan_validated"
        )}],
    }


def inspect_local_artifacts_node(state: AgentState) -> dict:
    paper_id = state.get("paper_id")
    if paper_id is None:
        return {
            "artifact_check": {
                "check_type": "existence_only_not_integrity_verified", "ready": False,
                "present": [], "missing": list(ARTIFACT_LABELS),
            },
            "trace": [{"node": "inspect_local_artifacts", "event": "paper_not_selected"}],
        }
    # Defense in depth if the node is tested or reused outside the graph.
    paper_id = validate_paper_id(paper_id)
    data_dir = Path(state.get("data_dir", "data"))
    targets = {
        "chunks": data_dir / "chunks" / f"{paper_id}.json",
        "embedding_vectors": data_dir / "embeddings" / f"{paper_id}.npy",
        "embedding_manifest": data_dir / "embeddings" / f"{paper_id}.manifest.json",
        "faiss_index": data_dir / "indexes" / f"{paper_id}.faiss",
        "index_manifest": data_dir / "indexes" / f"{paper_id}.manifest.json",
    }
    present = [name for name, path in targets.items() if path.is_file()]
    missing = [name for name in targets if name not in present]
    return {
        "artifact_check": {
            "check_type": "existence_only_not_integrity_verified",
            "ready": not missing,
            "present": present, "missing": missing,
        },
        "trace": [{"node": "inspect_local_artifacts", "event": "file_presence_checked"}],
    }


def route_after_inspection(state: AgentState) -> Literal["ready", "needs_paper", "missing"]:
    if state.get("paper_id") is None:
        return "needs_paper"
    if state.get("artifact_check", {}).get("ready"):
        return "ready"
    return "missing"


def ready_node(state: AgentState) -> dict:
    return {
        "status": "local_artifacts_present_unverified",
        "trace": [{"node": "ready", "event": "can_attempt_retrieval_later"}],
    }


def needs_paper_node(state: AgentState) -> dict:
    return {
        "status": "needs_paper_selection",
        "trace": [{"node": "needs_paper", "event": "no_paper_id"}],
    }


def missing_node(state: AgentState) -> dict:
    return {
        "status": "missing_local_artifacts",
        "trace": [{"node": "missing", "event": "offline_preflight_failed"}],
    }


def finalize_node(state: AgentState) -> dict:
    return {"trace": [{"node": "finalize", "event": "no_answer_generated"}]}


def build_research_graph(planner_provider: Callable[[str, list[str]], list[str]] | None = None):
    """Compile a genuine LangGraph StateGraph (optional 'agent' extra)."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise AgentDependencyError(
            'LangGraph is missing. Install with: python -m pip install -e ".[agent]"'
        ) from exc

    graph = StateGraph(AgentState)
    graph.add_node("validate_request", validate_request_node)
    graph.add_node("plan_research", lambda state: plan_research_node(state, planner_provider))
    graph.add_node("inspect_local_artifacts", inspect_local_artifacts_node)
    graph.add_node("ready", ready_node)
    graph.add_node("needs_paper", needs_paper_node)
    graph.add_node("missing", missing_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "validate_request")
    graph.add_conditional_edges(
        "validate_request", route_after_validation,
        {"plan": "plan_research", "finalize": "finalize"},
    )
    graph.add_edge("plan_research", "inspect_local_artifacts")
    graph.add_conditional_edges(
        "inspect_local_artifacts", route_after_inspection,
        {"ready": "ready", "needs_paper": "needs_paper", "missing": "missing"},
    )
    for node in ("ready", "needs_paper", "missing"):
        graph.add_edge(node, "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agent_preview(
    question: str, *, paper_id: str | None = None,
    data_dir: Path = Path("data"), max_papers: int = 5,
    planner_mode: Literal["deterministic", "llm"] = "deterministic",
    planner_api_key: str | None = None,
    planner_model: str = "gpt-4.1-mini",
    planner_base_url: str = "https://api.openai.com/v1",
    planner_timeout: float = 20.0,
) -> dict:
    """Run planning and artifact preflight; no research or answer generation."""
    request = ResearchRequest(question=question, max_papers=max_papers)
    if paper_id is not None:
        paper_id = validate_paper_id(paper_id)
    if planner_mode not in {"deterministic", "llm"}:
        raise PlanningError("Planner mode must be deterministic or llm")
    provider = None
    if planner_mode == "llm" and planner_api_key:
        from .llm_planner import OpenAICompatibleQuestionPlanner
        provider = OpenAICompatibleQuestionPlanner(
            api_key=planner_api_key, model=planner_model,
            base_url=planner_base_url, timeout=planner_timeout,
        )
    graph = build_research_graph(provider) if provider is not None else build_research_graph()
    output = graph.invoke({
        "question": request.question, "paper_id": paper_id,
        "data_dir": str(data_dir), "max_papers": request.max_papers,
        "planner_requested": planner_mode,
        "trace": [],
    }, config={"recursion_limit": 16})
    return {
        "status": output["status"],
        "milestone": "3.2",
        "engine": "langgraph_stategraph",
        "paper_id": output.get("paper_id"),
        "question_analysis": output.get("question_analysis"),
        "plan": output.get("plan", []),
        "plan_validation": output.get("plan_validation"),
        "planner": {
            "requested": planner_mode,
            "used": output.get("planner_used", "not_run"),
            "fallback_reason": output.get("planner_fallback_reason"),
            "model_output_is_unverified": output.get("planner_used") == "llm_proposal_unverified",
        },
        "artifact_check": output.get("artifact_check"),
        "trace": output.get("trace", []),
        "errors": output.get("errors", []),
        "tool_calls": 0,
        "llm_calls": output.get("llm_calls", 0),
        "research_executed": False,
        "answer_generated": False,
        "limitations": [
            "Only research planning; no search, PDF extraction, retrieval or answer synthesis.",
            "Proposed LLM subquestions are not verified facts, evidence, or tool calls.",
            "Local artifact checks verify file presence, not SHA-256 or page/source integrity.",
            "No persistent checkpoint or autonomous replanning is implemented.",
        ],
    }

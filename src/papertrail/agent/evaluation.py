"""Phase 4.1 offline pipeline diagnostics, not a validated research-quality benchmark.

Provisional reference chunk IDs are log-derived anchors. An unlisted retrieved
chunk is *unjudged*, not irrelevant. Never send requests to a language model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentEvaluationError(ValueError):
    """Unsafe, invalid, or unavailable evaluation input."""


class ReferenceFacet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str = Field(pattern=r"^retrieve_\d{2}$")
    anchor_chunk_ids: list[str] = Field(min_length=1, max_length=8)
    rationale: str = Field(min_length=10, max_length=300)

    @model_validator(mode="after")
    def distinct_anchors(self):
        if len(set(self.anchor_chunk_ids)) != len(self.anchor_chunk_ids):
            raise ValueError("Repeated anchor")
        if any(not 10 <= len(chunk_id) <= 300 or not re.fullmatch(r"[a-zA-Z0-9_@:.\-]+", chunk_id)
               for chunk_id in self.anchor_chunk_ids):
            raise ValueError("Invalid chunk reference")
        return self


class AgentEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,60}$")
    paper_id: str = Field(min_length=5, max_length=32)
    question: str = Field(min_length=12, max_length=1000)
    facets: list[ReferenceFacet] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def unique_steps(self):
        from papertrail.ingestion.pdf_downloader import validate_paper_id
        validate_paper_id(self.paper_id)
        if len({facet.step_id for facet in self.facets}) != len(self.facets):
            raise ValueError("Repeated facet")
        return self


class AgentEvaluationSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suite_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,60}$")
    annotation_status: str = Field(pattern=r"^provisional_log_derived_unverified$")
    provenance: str = Field(min_length=12, max_length=600)
    cases: list[AgentEvaluationCase] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_case_ids(self):
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("Repeated case ID")
        return self


def load_agent_suite(path: Path) -> AgentEvaluationSuite:
    try:
        if path.stat().st_size > 256_000:
            raise AgentEvaluationError("Evaluation suite exceeds 256 KB")
        return AgentEvaluationSuite.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise AgentEvaluationError("Cannot read or validate provisional evaluation suite") from exc


def parse_k_values(raw: str) -> tuple[int, ...]:
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+){0,3}", raw):
        raise AgentEvaluationError("--k-values must be 1-4 comma-separated integers")
    values = tuple(int(item) for item in raw.split(","))
    if values != tuple(sorted(set(values))) or not all(1 <= k <= 10 for k in values):
        raise AgentEvaluationError("--k-values must be unique, increasing, and in range 1..10")
    return values


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _safe_diagnostic_atom(value: object) -> str:
    """Only fixed-shape codes enter diagnostics; never expose tool payloads or secrets."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
        return "unknown"
    return value


def _preview_failure_diagnostic(result: dict, case: AgentEvaluationCase, top_k: int) -> str:
    """Return actionable failure metadata, not untrusted excerpts or raw exceptions."""
    status = _safe_diagnostic_atom(result.get("status"))
    stage, reason = "unknown", "unknown"
    events = result.get("tool_events")
    if isinstance(events, list):
        for item in reversed(events):
            if not isinstance(item, dict) or item.get("status") != "failed":
                continue
            stage = _safe_diagnostic_atom(item.get("name"))
            detail = item.get("detail")
            if isinstance(detail, dict):
                reason = _safe_diagnostic_atom(detail.get("error_type"))
            break
    trace = result.get("trace")
    last_node = "unknown"
    if isinstance(trace, list):
        for item in reversed(trace):
            if isinstance(item, dict) and isinstance(item.get("node"), str):
                last_node = _safe_diagnostic_atom(item["node"])
                break
    label = _safe_diagnostic_atom(case.case_id)
    return (f"Offline preview failed: case={label}; top_k={top_k}; status={status}; "
            f"failed_tool={stage}; error_type={reason}; last_node={last_node}. "
            "No LLM call was made. Run agent-answer --mode preview with the same paper/question/top-k "
            "and inspect status, errors, and tool_events; do not enable --allow-network.")


def evaluate_preview_run(result: dict, case: AgentEvaluationCase, top_k: int) -> dict:
    """Compare logged preview retrieval and model-visible register with known anchors.

    Exact chunk IDs only. The denominator consists of *provisional anchors*,
    not all relevant passages in the PDF. No precision/factuality assertions.
    """
    if result.get("paper_id") != case.paper_id or result.get("llm_calls") != 0 or result.get("network_opt_in") is not False:
        raise AgentEvaluationError("Offline evaluation invariant violated")
    if result.get("generation_mode") != "preview" or result.get("answer_generated") is not False:
        raise AgentEvaluationError("Evaluation requires non-generating preview only")
    if result.get("status") not in {"ready_for_llm_synthesis", "insufficient_evidence"}:
        raise AgentEvaluationError(_preview_failure_diagnostic(result, case, top_k))
    evidence = result.get("evidence")
    synth = result.get("synthesis")
    if not isinstance(evidence, list) or not isinstance(synth, dict):
        raise AgentEvaluationError("Preview output is missing evidence fields")
    register = synth.get("source_register", [])
    if not isinstance(register, list):
        raise AgentEvaluationError("Preview output is missing the selected evidence register")
    if not register and result["status"] == "ready_for_llm_synthesis":
        raise AgentEvaluationError("Ready preview has no evidence register")
    retrieved: dict[str, list[str]] = {}
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("hits"), list):
            raise AgentEvaluationError("Invalid retrieved evidence")
        step_id = item.get("step_id")
        if step_id in retrieved:
            retrieved[step_id] += [str(hit["chunk_id"]) for hit in item["hits"]]
        else:
            retrieved[step_id] = [str(hit["chunk_id"]) for hit in item["hits"]]
    selected: dict[str, list[str]] = {}
    for entry in register:
        selected.setdefault(entry["step_id"], []).append(entry["chunk_id"])
    total_refs = in_retrieval = in_register = facet_hit = facet_register_hit = 0
    rows = []
    for facet in case.facets:
        anchors = facet.anchor_chunk_ids
        found = [a for a in anchors if a in retrieved.get(facet.step_id, [])]
        selected_anchors = [a for a in anchors if a in selected.get(facet.step_id, [])]
        if not set(selected_anchors).issubset(set(found)):
            raise AgentEvaluationError("Selected source is not from retrieved primary evidence")
        total_refs += len(anchors)
        in_retrieval += len(found)
        in_register += len(selected_anchors)
        facet_hit += bool(found)
        facet_register_hit += bool(selected_anchors)
        rows.append({
            "step_id": facet.step_id,
            "reference_anchor_ids": anchors,
            "retrieved_reference_ids": found,
            "model_visible_reference_ids": selected_anchors,
            "not_retrieved_ids": [a for a in anchors if a not in found],
            "retrieved_but_not_selected_ids": [a for a in found if a not in selected_anchors],
            "primary_hit_count": len(retrieved.get(facet.step_id, [])),
            "registered_hit_count": len(selected.get(facet.step_id, [])),
            "observed_anchor_retrieval_fraction": round(len(found) / len(anchors), 4),
            "observed_anchor_model_visible_fraction": round(len(selected_anchors) / len(anchors), 4),
        })
    all_hits = [id_ for ids in retrieved.values() for id_ in ids]
    all_selected = [id_ for ids in selected.values() for id_ in ids]
    return {
        "case_id": case.case_id,
        "paper_id": case.paper_id,
        "top_k": top_k,
        "preview_status": result["status"],
        "facet_count": len(case.facets),
        "reference_anchor_count": total_refs,
        "anchors_retrieved": in_retrieval,
        "anchors_model_visible": in_register,
        "facets_with_observed_retrieval_anchor": facet_hit,
        "facets_with_model_visible_anchor": facet_register_hit,
        "observed_anchor_retrieval_fraction": round(in_retrieval / total_refs, 4),
        "observed_anchor_model_visible_fraction": round(in_register / total_refs, 4),
        "primary_hits": len(all_hits),
        "unique_primary_chunks": len(set(all_hits)),
        "selected_register_entries": len(all_selected),
        "unique_model_visible_chunks": len(set(all_selected)),
        "llm_calls": 0,
        "facets": rows,
    }


def load_recorded_live_snapshot(path: Path, paper_id: str) -> dict:
    """Read a sanitized *observed* live summary; never infer claim entailment."""
    try:
        if path.stat().st_size > 16_000:
            raise AgentEvaluationError("Recorded live summary exceeds 16 KB")
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AgentEvaluationError("Cannot read or validate recorded live summary") from exc
    expected = {"kind", "paper_id", "provenance", "status", "llm_calls", "accepted_claims",
                "rejected_claims", "covered_step_ids", "missing_step_ids", "semantic_entailment_verified"}
    if (not isinstance(obj, dict) or set(obj) != expected or obj.get("kind") != "observed_live_summary_unverified"
            or obj.get("paper_id") != paper_id or obj.get("status") not in
            {"cited_answer_unverified", "partial_answer_unverified", "insufficient_evidence", "citation_validation_failed"}
            or not isinstance(obj.get("provenance"), str) or len(obj["provenance"]) > 500
            or type(obj.get("llm_calls")) is not int or not 0 <= obj["llm_calls"] <= 10
            or type(obj.get("accepted_claims")) is not int or not 0 <= obj["accepted_claims"] <= 30
            or type(obj.get("rejected_claims")) is not int or not 0 <= obj["rejected_claims"] <= 30
            or type(obj.get("semantic_entailment_verified")) is not bool or obj["semantic_entailment_verified"]
            or not isinstance(obj.get("covered_step_ids"), list) or not isinstance(obj.get("missing_step_ids"), list)
            or any(not isinstance(x, str) or not re.fullmatch(r"retrieve_\d{2}", x)
                   for x in obj["covered_step_ids"] + obj["missing_step_ids"])
            or set(obj["covered_step_ids"]) & set(obj["missing_step_ids"])
            or len(set(obj["covered_step_ids"] + obj["missing_step_ids"])) != len(obj["covered_step_ids"] + obj["missing_step_ids"])):
        raise AgentEvaluationError("Recorded live summary is invalid or has unverifiable status")
    total = obj["accepted_claims"] + obj["rejected_claims"]
    return {**obj, "observed_claim_acceptance_fraction": round(obj["accepted_claims"] / total, 4) if total else None,
            "observed_facet_coverage_fraction": round(len(obj["covered_step_ids"]) /
                                                       len(obj["covered_step_ids"] + obj["missing_step_ids"]), 4)
            if (obj["covered_step_ids"] or obj["missing_step_ids"]) else None,
            "independent_claim_validation_performed": False}


def run_agent_evaluation(
    suite: AgentEvaluationSuite, *, data_dir: Path, k_values: tuple[int, ...] = (3, 5),
    runner: Callable[[str, str, int], dict] | None = None,
    recorded_live: Path | None = None,
) -> dict:
    """Compare two or more bounded local preview runs; no LLM or arXiv network."""
    if not k_values or len(k_values) > 4 or tuple(sorted(set(k_values))) != k_values or any(not 1 <= k <= 10 for k in k_values):
        raise AgentEvaluationError("Invalid top-k configuration")
    if runner is None:
        from papertrail.agent.tool_execution import ExecutionOptions, run_agent_research

        def run_default(question: str, paper_id: str, k: int) -> dict:
            return run_agent_research(
                question, paper_id=paper_id,
                options=ExecutionOptions(data_dir=data_dir, top_k=k, allow_network=False,
                                         auto_select_first=False, max_refinement_rounds=0),
                planner_mode="deterministic", synthesize_mode=True, synthesis_provider=None,
            )
        runner = run_default
    records = []
    for case in suite.cases:
        for top_k in k_values:
            result = runner(case.question, case.paper_id, top_k)
            records.append(evaluate_preview_run(result, case, top_k))
    replay = None
    if recorded_live is not None:
        if len({c.paper_id for c in suite.cases}) != 1:
            raise AgentEvaluationError("Live summary comparison requires a single-paper suite")
        replay = load_recorded_live_snapshot(recorded_live, suite.cases[0].paper_id)
    return {
        "phase": "4.1", "status": "provisional_offline_agent_diagnostics_complete",
        "suite_id": suite.suite_id, "annotation_status": suite.annotation_status,
        "independently_labeled_ground_truth": False, "semantic_entailment_verified": False,
        "provenance": suite.provenance,
        "llm_calls": 0, "network_opt_in": False,
        "configs": list(k_values), "case_count": len(suite.cases), "run_count": len(records),
        "per_run": records, "recorded_live_observation": replay,
        "interpretation": (
            "Reference chunk IDs are provisional observed anchors from previous local runs, not exhaustive qrels. "
            "Fractions are ONLY observed-anchor coverage, not scientific retrieval recall/precision. "
            "Model-visible anchors indicate source selection, not semantic sufficiency. "
            "The live summary is historical reported output, not a new model call or independent validation."
        ),
    }


def save_agent_evaluation(report: dict, data_dir: Path) -> tuple[Path, Path]:
    suite_id = report.get("suite_id", "")
    if not isinstance(suite_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,60}", suite_id):
        raise AgentEvaluationError("Invalid suite name")
    folder = data_dir / "agent_evaluations"
    folder.mkdir(parents=True, exist_ok=True)
    path_json = folder / f"{suite_id}_phase4_1.json"
    path_md = folder / f"{suite_id}_phase4_1.md"
    path_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PaperTrail Agent Diagnostics — Phase 4.1", "", "**Provisional, log-derived anchors; NOT independently judged qrels.**", "",
             "| Case | top-k | Anchor retrieved | Anchor model-visible | Primary hits | Register entries |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in report["per_run"]:
        lines.append(f"| {row['case_id']} | {row['top_k']} | {row['anchors_retrieved']}/{row['reference_anchor_count']} | "
                     f"{row['anchors_model_visible']}/{row['reference_anchor_count']} | "
                     f"{row['primary_hits']} | {row['selected_register_entries']} |")
    lines.extend(["", "## Per-facet diagnostics", ""])
    for row in report["per_run"]:
        lines.append(f"### {row['case_id']} — top-k={row['top_k']}")
        for facet in row["facets"]:
            lines.append(f"- **{facet['step_id']}**: missing in retrieval `{', '.join(facet['not_retrieved_ids']) or 'none'}`; "
                         f"retrieved but not selected `{', '.join(facet['retrieved_but_not_selected_ids']) or 'none'}`")
        lines.append("")
    if report["recorded_live_observation"]:
        replay = report["recorded_live_observation"]
        lines.extend(["## Historical Live (reported only)", "",
                      f"- Reported status: `{replay['status']}`",
                      f"- Accepted/rejected claims: {replay['accepted_claims']}/{replay['rejected_claims']}",
                      f"- Facets covered: {', '.join(replay['covered_step_ids']) or 'none'}",
                      f"- Missing facets: {', '.join(replay['missing_step_ids']) or 'none'}", ""])
    lines.extend(["## Limitations", "", report["interpretation"],
                  "", "Requires external human-verified labels and claim-level entailment review before any scientific quality claim.", ""])
    path_md.write_text("\n".join(lines), encoding="utf-8")
    return path_json, path_md

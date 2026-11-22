"""Phase 4.2: paired, local-only baseline/focused query strategy diagnostics.

The supplied four source anchors are provisional and non-exhaustive. These
numbers are *not* verified Recall@K, MRR or semantic relevance. Source chunks
outside the examples are unjudged. No LLM call, Internet or policy deployment.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from .evaluation import (AgentEvaluationError, AgentEvaluationSuite,
                         evaluate_preview_run)
from .query_rewriting import focused_queries
from .planning import analyze_question


STYLES = ("baseline", "focused")


def _facet_diagnostics(result: dict, case, baseline_row: dict) -> list[dict]:
    observed_hits: dict[str, list[str]] = {}
    for item in result["evidence"]:
        observed_hits.setdefault(item["step_id"], []).extend(hit["chunk_id"] for hit in item["hits"])
    selected = result["synthesis"]["source_register"]
    queries = {step["id"]: step["search_query"] for step in result["plan"]
               if step["proposed_tool"] == "hybrid_retrieval"}
    rows = []
    for facet, existing in zip(case.facets, baseline_row["facets"], strict=True):
        ranked = observed_hits.get(facet.step_id, [])
        first_rank = next((rank for rank, name in enumerate(ranked, start=1)
                           if name in facet.anchor_chunk_ids), None)
        in_register = {entry["chunk_id"] for entry in selected if entry["step_id"] == facet.step_id}
        rows.append({
            **existing,
            "query": queries.get(facet.step_id),
            "first_observed_anchor_rank": first_rank,
            "observed_first_anchor_reciprocal_rank": round(1 / first_rank, 4) if first_rank else 0.,
            "observed_anchor_in_register": any(item in in_register for item in facet.anchor_chunk_ids),
        })
    return rows


def evaluate_query_preview(result: dict, case, top_k: int, style: str) -> dict:
    if style not in STYLES or result.get("query_style") != style:
        raise AgentEvaluationError("Query strategy mismatch: cannot compare different execution paths")
    record = evaluate_preview_run(result, case, top_k)
    rows = _facet_diagnostics(result, case, record)
    if any(f["query"] is None for f in rows):
        raise AgentEvaluationError("Preview missing retrieval query audit trail")
    if style == "focused":
        intents = analyze_question(case.question, paper_id=case.paper_id).intents
        expected = focused_queries(case.question, intents)
        if len(expected) != len(rows) or [item["query"] for item in rows] != expected:
            raise AgentEvaluationError("Preview focused strategy disagrees with declared generator")
    return {
        **record, "query_style": style, "facets": rows,
        "observed_facet_anchor_mrr": round(sum(f["observed_first_anchor_reciprocal_rank"] for f in rows) / len(rows), 4),
        "query_strategy_audited": True,
    }


def run_query_evaluation(
    suite: AgentEvaluationSuite, *, data_dir: Path, k_values: tuple[int, ...] = (3, 5),
    runner: Callable[[str, str, int, str], dict] | None = None,
) -> dict:
    """Paired comparisons on identical case/paper/k/retriever; no automatic winner."""
    if not k_values or len(k_values) > 4 or k_values != tuple(sorted(set(k_values))) or any(
        type(k) is not int or not 1 <= k <= 10 for k in k_values
    ):
        raise AgentEvaluationError("Invalid top-k configuration")
    if runner is None:
        from papertrail.agent.tool_execution import ExecutionOptions, run_agent_research

        def local_runner(question: str, paper_id: str, k: int, style: str) -> dict:
            return run_agent_research(
                question, paper_id=paper_id,
                options=ExecutionOptions(data_dir=data_dir, top_k=k, query_style=style,
                                         allow_network=False, auto_select_first=False,
                                         max_refinement_rounds=0),
                planner_mode="deterministic", synthesize_mode=True, synthesis_provider=None,
            )
        runner = local_runner
    runs = []
    comparisons = []
    for case in suite.cases:
        for k in k_values:
            pair = {}
            for style in STYLES:
                result = runner(case.question, case.paper_id, k, style)
                record = evaluate_query_preview(result, case, k, style)
                pair[style] = record
                runs.append(record)
            a, b = pair["baseline"], pair["focused"]
            comparisons.append({
                "case_id": case.case_id, "paper_id": case.paper_id, "top_k": k,
                "baseline_observed_anchor_fraction": a["observed_anchor_retrieval_fraction"],
                "focused_observed_anchor_fraction": b["observed_anchor_retrieval_fraction"],
                "delta_observed_anchor_fraction": round(b["observed_anchor_retrieval_fraction"] - a["observed_anchor_retrieval_fraction"], 4),
                "baseline_model_visible_anchor_fraction": a["observed_anchor_model_visible_fraction"],
                "focused_model_visible_anchor_fraction": b["observed_anchor_model_visible_fraction"],
                "delta_model_visible_fraction": round(b["observed_anchor_model_visible_fraction"] - a["observed_anchor_model_visible_fraction"], 4),
                "baseline_observed_facet_mrr": a["observed_facet_anchor_mrr"],
                "focused_observed_facet_mrr": b["observed_facet_anchor_mrr"],
                "delta_observed_facet_mrr": round(b["observed_facet_anchor_mrr"] - a["observed_facet_anchor_mrr"], 4),
            })
    return {
        "phase": "4.2", "status": "provisional_paired_query_diagnostics_complete",
        "suite_id": suite.suite_id, "annotation_status": suite.annotation_status,
        "independently_labeled_ground_truth": False,
        "semantic_entailment_verified": False, "default_agent_query_style_changed": False,
        "llm_calls": 0, "network_opt_in": False,
        "k_values": list(k_values), "styles": list(STYLES),
        "case_count": len(suite.cases), "run_count": len(runs),
        "per_run": runs, "comparisons": comparisons,
        "interpretation": (
            "Exploratory paired checks against selected log-derived, non-exhaustive chunk anchors; "
            "all other retrieved passages remain UNJUDGED. Observed-anchor hit fractions and first-anchor "
            "reciprocal rank are NOT gold Recall@K, MRR, precision or correctness. "
            "Same local retrieval backend and top-k are used for each pair. "
            "Neither a single paper nor post-hoc query terms provide held-out evidence of generalization. "
            "Any deployment choice requires independent human labels, multi-paper holdout and semantic review."
        ),
    }


def save_query_evaluation(report: dict, data_dir: Path) -> tuple[Path, Path]:
    name = report.get("suite_id")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,60}", name):
        raise AgentEvaluationError("Invalid suite ID")
    folder = data_dir / "agent_evaluations"
    folder.mkdir(parents=True, exist_ok=True)
    dest_json = folder / f"{name}_phase4_2_query_comparison.json"
    dest_md = folder / f"{name}_phase4_2_query_comparison.md"
    dest_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Phase 4.2 — Query-strategy A/B (diagnostic only)", "",
             "**Unverified, non-exhaustive log-derived anchors — NOT scientific retrieval metrics.**", "",
             "| Case | k | Baseline anchors | Focused anchors | Δ observed-anchor fraction | Baseline facet RR | Focused facet RR |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for c in report["comparisons"]:
        lines.append(f"| {c['case_id']} | {c['top_k']} | {c['baseline_observed_anchor_fraction']:.4f} | "
                     f"{c['focused_observed_anchor_fraction']:.4f} | {c['delta_observed_anchor_fraction']:+.4f} | "
                     f"{c['baseline_observed_facet_mrr']:.4f} | {c['focused_observed_facet_mrr']:.4f} |")
    lines.extend(["", "## Per-facet query traces", ""])
    for row in report["per_run"]:
        lines.append(f"### {row['case_id']} | k={row['top_k']} | {row['query_style']}")
        for f in row["facets"]:
            lines.extend([f"- **{f['step_id']}** search: `{f['query']}`",
                          f"  - Observed anchor rank: `{f['first_observed_anchor_rank']}`; "
                          f"found {len(f['retrieved_reference_ids'])}/{len(f['reference_anchor_ids'])}; "
                          f"visible {len(f['model_visible_reference_ids'])}/{len(f['reference_anchor_ids'])}"])
        lines.append("")
    lines.extend(["## Interpretation and caveats", "", report["interpretation"], "",
                  "No new API calls. Focused is opt-in and has NOT replaced the baseline.", ""])
    dest_md.write_text("\n".join(lines), encoding="utf-8")
    return dest_json, dest_md

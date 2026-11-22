"""Phase 4.2.2: paired offline comparison of RRF vs budget-aware interleaving.

Four provisional log-derived anchors are diagnostics, not gold relevance labels.
No preview or eval here may consult those anchors to select a retrieval policy.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from .evaluation import AgentEvaluationError, AgentEvaluationSuite
from .multi_eval import evaluate_multi_preview
from .query_eval import evaluate_query_preview

STYLES = ("baseline", "focused", "multi", "budgeted")


def run_budget_evaluation(
    suite: AgentEvaluationSuite, *, data_dir: Path, k_values: tuple[int, ...] = (3, 5),
    runner: Callable[[str, str, int, str], dict] | None = None,
) -> dict:
    if (not k_values or len(k_values) > 4 or k_values != tuple(sorted(set(k_values)))
            or any(type(k) is not int or not 1 <= k <= 10 for k in k_values)):
        raise AgentEvaluationError("Invalid fixed-budget comparison top-k values")
    if runner is None:
        from .tool_execution import ExecutionOptions, run_agent_research

        def default_runner(question: str, paper_id: str, k: int, style: str) -> dict:
            return run_agent_research(
                question, paper_id=paper_id,
                options=ExecutionOptions(data_dir=data_dir, top_k=k, query_style=style,
                                         allow_network=False, auto_select_first=False,
                                         max_refinement_rounds=0),
                planner_mode="deterministic", synthesize_mode=True, synthesis_provider=None,
            )
        runner = default_runner
    per_run: list[dict] = []
    comparisons: list[dict] = []
    for case in suite.cases:
        for top_k in k_values:
            rows = {}
            results = {}
            for style in STYLES:
                result = runner(case.question, case.paper_id, top_k, style)
                row = (evaluate_multi_preview(result, case, top_k, style=style)
                       if style in {"multi", "budgeted"}
                       else evaluate_query_preview(result, case, top_k, style))
                results[style], rows[style] = result, row
                per_run.append(row)
            baseline, focused, multi, budgeted = (rows[style] for style in STYLES)
            union_count = 0
            lost = {"multi": [], "budgeted": []}
            dropped_register = {"multi": [], "budgeted": []}
            for idx, facet in enumerate(case.facets):
                expected = {
                    "baseline": [hit["chunk_id"] for hit in results["baseline"]["evidence"][idx]["hits"]],
                    "focused": [hit["chunk_id"] for hit in results["focused"]["evidence"][idx]["hits"]],
                }
                union = set(expected["baseline"]) | set(expected["focused"])
                anchor_union = set(expected["baseline"]) | set(expected["focused"])
                union_count += sum(anchor in anchor_union for anchor in facet.anchor_chunk_ids)
                for style in ("multi", "budgeted"):
                    trace = results[style]["evidence"][idx]["query_traces"]
                    if (trace[0]["chunk_ids"] != expected["baseline"] or
                            trace[1]["chunk_ids"] != expected["focused"]):
                        raise AgentEvaluationError("Paired branch results differ: cannot compare fusion")
                    actual = {hit["chunk_id"] for hit in results[style]["evidence"][idx]["hits"]}
                    if not actual.issubset(union):
                        raise AgentEvaluationError("Fusion introduced a non-source chunk")
                    details = rows[style]["facet_diagnostics"][idx]
                    lost[style] += [{"step_id": facet.step_id, "chunk_id": ch}
                                    for ch in details["reference_ids_missing_after_fusion"]]
                    dropped_register[style] += [{"step_id": facet.step_id, "chunk_id": ch}
                                                for ch in details["reference_ids_missing_after_register"]]
            refs = baseline["reference_anchor_count"]
            comparisons.append({
                "case_id": case.case_id, "paper_id": case.paper_id, "top_k": top_k,
                "total_provisional_anchors": refs,
                "baseline_anchors": baseline["anchors_retrieved"],
                "focused_anchors": focused["anchors_retrieved"],
                "union_anchors_pre_budget_theoretical": union_count,
                "rrf_anchors_post_budget": multi["anchors_retrieved"],
                "budgeted_anchors_post_budget": budgeted["anchors_retrieved"],
                "rrf_anchors_model_visible": multi["anchors_model_visible"],
                "budgeted_anchors_model_visible": budgeted["anchors_model_visible"],
                "rrf_observed_anchor_fraction": multi["observed_anchor_retrieval_fraction"],
                "budgeted_observed_anchor_fraction": budgeted["observed_anchor_retrieval_fraction"],
                "delta_budgeted_vs_rrf_fraction": round(
                    budgeted["observed_anchor_retrieval_fraction"] - multi["observed_anchor_retrieval_fraction"], 4),
                "lost_to_fusion": lost,
                "lost_to_source_register": dropped_register,
                "retrieval_query_counts": {"baseline": len(case.facets), "focused": len(case.facets),
                                           "multi": multi["query_tool_calls"],
                                           "budgeted": budgeted["query_tool_calls"]},
            })
    return {
        "phase": "4.2.2", "status": "provisional_budget_fusion_diagnostics_complete",
        "suite_id": suite.suite_id, "annotation_status": suite.annotation_status,
        "independently_labeled_ground_truth": False, "semantic_entailment_verified": False,
        "default_agent_query_style_changed": False, "llm_calls": 0, "network_opt_in": False,
        "strategies": list(STYLES), "k_values": list(k_values),
        "case_count": len(suite.cases), "run_count": len(per_run),
        "per_run": per_run, "comparisons": comparisons,
        "interpretation": (
            "Exploratory same-paper paired fusion ablation based on SELECTED, NON-EXHAUSTIVE, "
            "UNVERIFIED log-derived anchors. Non-anchor chunks are UNJUDGED, not negatives. "
            "Observed-anchor coverage is NOT gold Recall@K or validated answer faithfulness. "
            "An oracle union is only a pre-budget diagnostic. Both multi and budgeted execute "
            "two local queries per facet, so they cost twice baseline retrieval calls. "
            "The budgeted policy is focused-first interleaving, not semantic reranking; it can "
            "lose evidence and is not proven superior. No live LLM or network calls. "
            "Human-judged multi-paper holdout is required for a deployment decision."
        ),
    }


def save_budget_evaluation(report: dict, data_dir: Path) -> tuple[Path, Path]:
    name = report.get("suite_id")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,60}", name):
        raise AgentEvaluationError("Invalid suite ID")
    target = data_dir / "agent_evaluations"
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / f"{name}_phase4_2_2_budget_fusion.json"
    markdown_path = target / f"{name}_phase4_2_2_budget_fusion.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Phase 4.2.2 — Budget-aware fusion A/B (provisional)", "",
             "**Four unverified log-derived anchors; these are NOT gold retrieval metrics.**", "",
             "| Case | k | Baseline | Focused | RRF post-budget | Budget-aware post-budget | Budget-aware model visible | Union (theoretical) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in report["comparisons"]:
        total = c["total_provisional_anchors"]
        lines.append(f"| {c['case_id']} | {c['top_k']} | {c['baseline_anchors']}/{total} | "
                     f"{c['focused_anchors']}/{total} | {c['rrf_anchors_post_budget']}/{total} | "
                     f"{c['budgeted_anchors_post_budget']}/{total} | "
                     f"{c['budgeted_anchors_model_visible']}/{total} | "
                     f"{c['union_anchors_pre_budget_theoretical']}/{total} |")
        lines += ["", f"## {c['case_id']} | top-k={c['top_k']}",
                  f"- Anchor delta vs RRF: {c['delta_budgeted_vs_rrf_fraction']:+.4f}",
                  f"- Retrieval calls per run: {c['retrieval_query_counts']}",
                  f"- RRF lost before register: {c['lost_to_fusion']['multi']}",
                  f"- Budget-aware lost before register: {c['lost_to_fusion']['budgeted']}",
                  f"- Budget-aware dropped by register: {c['lost_to_source_register']['budgeted']}"]
        row = next(r for r in report["per_run"] if r["case_id"] == c["case_id"] and
                   r["top_k"] == c["top_k"] and r["query_style"] == "budgeted")
        for facet in row["facet_diagnostics"]:
            lines += ["", f"### {facet['step_id']}",
                      f"- Search queries: `{facet['baseline_query']}` AND `{facet['focused_query']}`",
                      f"- Primary hits baseline/focused: {facet['branch_hits']}",
                      f"- Unique primary pre-budget: {facet['union_primary_chunks_pre_budget']}; "
                      f"post-budget: {facet['merged_primary_chunks_post_budget']}",
                      f"- First provisional anchor rank: {facet['first_observed_anchor_rank']}",
                      f"- Missing provisional anchors after fusion: {facet['reference_ids_missing_after_fusion']}"]
        lines.append("")
    lines += ["## Interpretation", "", report["interpretation"], "",
              "Production baseline remains unchanged. No LLM/network was used.", ""]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path

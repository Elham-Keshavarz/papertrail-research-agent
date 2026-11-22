"""Phase 4.2.1 offline three-way multi-query, fixed-budget diagnostics.

Historical observed anchors, not exhaustive human-adjudicated qrels.
RRF fusion is evaluated as executed, never replaced by the oracle union.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from .evaluation import AgentEvaluationError, AgentEvaluationSuite, evaluate_preview_run
from .query_eval import evaluate_query_preview
from .query_rewriting import focused_queries
from .planning import analyze_question, build_research_plan


STYLES = ("baseline", "focused", "multi")


def evaluate_multi_preview(result: dict, case, top_k: int, *, style: str = "multi") -> dict:
    """Verify branch audit & fixed final budget before scoring observed anchors."""
    if style not in {"multi", "budgeted"} or result.get("query_style") != style:
        raise AgentEvaluationError("Multi-query preview strategy mismatch")
    base = evaluate_preview_run(result, case, top_k)
    evidence = result.get("evidence")
    plan = build_research_plan(case.question, paper_id=case.paper_id)
    baseline = [s.search_query for s in plan.steps if s.proposed_tool == "hybrid_retrieval"]
    focused = focused_queries(case.question, analyze_question(case.question, paper_id=case.paper_id).intents)
    if len(evidence) != len(case.facets) or len(baseline) != len(case.facets):
        raise AgentEvaluationError("Multi-query facet count does not match the plan")
    details = []
    for idx, (entry, facet) in enumerate(zip(evidence, case.facets, strict=True)):
        if entry.get("step_id") != facet.step_id or entry.get("query") != baseline[idx]:
            raise AgentEvaluationError("Multi-query facet is not baseline-plan aligned")
        traces = entry.get("query_traces")
        if not isinstance(traces, list) or len(traces) != 2:
            raise AgentEvaluationError("Multi-query must audit exactly two retrieval branches")
        expected = (("baseline", baseline[idx]), ("focused", focused[idx]))
        for t, (branch_style, query) in zip(traces, expected, strict=True):
            if (not isinstance(t, dict) or t.get("style") != branch_style or t.get("query") != query
                    or not isinstance(t.get("chunk_ids"), list) or len(t["chunk_ids"]) > top_k
                    or len(set(t["chunk_ids"])) != len(t["chunk_ids"])
                    or any(not isinstance(id_, str) or not id_ for id_ in t["chunk_ids"])):
                raise AgentEvaluationError("Multi-query source trace malformed")
        union = set(traces[0]["chunk_ids"]) | set(traces[1]["chunk_ids"])
        actual = [item["chunk_id"] for item in entry["hits"]]
        if len(actual) > top_k or len(set(actual)) != len(actual) or not set(actual).issubset(union):
            raise AgentEvaluationError("Multi-query fused evidence not a bounded source subset")
        details.append({"step_id": facet.step_id,
                        "baseline_query": baseline[idx], "focused_query": focused[idx],
                        "branch_hits": [len(t["chunk_ids"]) for t in traces],
                        "union_primary_chunks_pre_budget": len(union),
                        "merged_primary_chunks_post_budget": len(actual),
                        "reference_ids_in_union_pre_budget": [a for a in facet.anchor_chunk_ids if a in union],
                        "reference_ids_missing_after_fusion": [a for a in facet.anchor_chunk_ids if a in union and a not in actual],
                        "reference_ids_missing_after_register": [a for a in facet.anchor_chunk_ids
                                                                 if a in actual and a not in base["facets"][idx]["model_visible_reference_ids"]],
                        "first_observed_anchor_rank": next((rank for rank, a in enumerate(actual, start=1)
                                                            if a in facet.anchor_chunk_ids), None),
                        })
    observed_calls = [e for e in result.get("tool_events", []) if e.get("name") == "hybrid_retrieval"
                      and e.get("status") == "success"]
    if len(observed_calls) != 2 * len(case.facets) or any(
        event.get("detail", {}).get("style") != STYLES[idx % 2] or
        event.get("detail", {}).get("step_id") != case.facets[idx // 2].step_id
        for idx, event in enumerate(observed_calls)
    ):
        # styles[0:2] is baseline/focused; multi is not a source query.
        raise AgentEvaluationError("Multi-query must perform two audited calls per facet")
    return {**base, "case_id": case.case_id, "top_k": top_k, "query_style": style, "fusion_algorithm": ("cross_query_rrf_k60_fixed_budget" if style == "multi" else "focused_first_rank_interleave_fixed_budget"),
            "query_strategy_audited": True, "query_tool_calls": len(observed_calls),
            "facet_diagnostics": details}


def run_multi_query_evaluation(
    suite: AgentEvaluationSuite, *, data_dir: Path, k_values: tuple[int, ...] = (3, 5),
    runner: Callable[[str, str, int, str], dict] | None = None,
) -> dict:
    if (not k_values or len(k_values) > 4 or k_values != tuple(sorted(set(k_values)))
            or any(type(k) is not int or not 1 <= k <= 10 for k in k_values)):
        raise AgentEvaluationError("Invalid top-k configuration")
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
    records = []
    comparisons = []
    for case in suite.cases:
        for top_k in k_values:
            triplet = {}
            for style in STYLES:
                result = runner(case.question, case.paper_id, top_k, style)
                row = (evaluate_multi_preview(result, case, top_k) if style == "multi"
                       else evaluate_query_preview(result, case, top_k, style))
                triplet[style] = row
                records.append(row)
            baseline, focused, multi = (triplet[style] for style in STYLES)
            union_count = 0
            lost_to_fusion = []
            for facet, a, b, multi_f in zip(case.facets, baseline["facets"], focused["facets"],
                                            multi["facet_diagnostics"], strict=True):
                observed_union = set(a["retrieved_reference_ids"]) | set(b["retrieved_reference_ids"])
                union_count += len(observed_union)
                # A cross-check against the actual two branch traces, not fabricated oracle output.
                if observed_union != set(multi_f["reference_ids_in_union_pre_budget"]):
                    raise AgentEvaluationError("Multi-query branches differ from the paired reference runs")
                lost_to_fusion += [{"step_id": facet.step_id, "chunk_id": anchor}
                                   for anchor in multi_f["reference_ids_missing_after_fusion"]]
            refs = baseline["reference_anchor_count"]
            comparisons.append({
                "case_id": case.case_id, "paper_id": case.paper_id, "top_k": top_k,
                "baseline_anchors": baseline["anchors_retrieved"],
                "focused_anchors": focused["anchors_retrieved"],
                "union_anchors_pre_budget_theoretical": union_count,
                "multi_anchors_post_budget": multi["anchors_retrieved"],
                "multi_anchors_model_visible": multi["anchors_model_visible"],
                "total_provisional_anchors": refs,
                "baseline_fraction": baseline["observed_anchor_retrieval_fraction"],
                "focused_fraction": focused["observed_anchor_retrieval_fraction"],
                "multi_fraction": multi["observed_anchor_retrieval_fraction"],
                "union_fraction_theoretical": round(union_count / refs, 4),
                "multi_visible_fraction": multi["observed_anchor_model_visible_fraction"],
                "lost_to_fusion": lost_to_fusion,
                "retrieval_query_counts": {"baseline": len(case.facets),
                                           "focused": len(case.facets), "multi": multi["query_tool_calls"]},
            })
    return {
        "phase": "4.2.1", "status": "provisional_multi_query_diagnostics_complete",
        "suite_id": suite.suite_id, "annotation_status": suite.annotation_status,
        "independently_labeled_ground_truth": False, "semantic_entailment_verified": False,
        "default_agent_query_style_changed": False, "llm_calls": 0, "network_opt_in": False,
        "strategies": list(STYLES), "fusion_algorithm": "cross_query_rrf_k60_fixed_budget",
        "k_values": list(k_values), "case_count": len(suite.cases), "run_count": len(records),
        "per_run": records, "comparisons": comparisons,
        "interpretation": (
            "Single-paper exploratory query ablation using FOUR selected, non-exhaustive, log-derived chunk anchors. "
            "All non-anchor chunks are UNJUDGED, not negatives. Union is a theoretical pre-budget diagnostic "
            "only; actual multi retrieval is explicitly capped to top-k per facet and a smaller source register. "
            "Fraction is NOT gold Recall@K, scientific MRR, answer faithfulness, or generalization. "
            "Multi runs twice as many local retrieval queries; no LLM or arXiv access. "
            "Human-adjudicated multiple-paper held-out qrels are required before deployment."
        ),
    }


def save_multi_query_evaluation(report: dict, data_dir: Path) -> tuple[Path, Path]:
    name = report.get("suite_id")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,60}", name):
        raise AgentEvaluationError("Invalid suite ID")
    folder = data_dir / "agent_evaluations"
    folder.mkdir(parents=True, exist_ok=True)
    jpath = folder / f"{name}_phase4_2_1_multi_query.json"
    mpath = folder / f"{name}_phase4_2_1_multi_query.md"
    jpath.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Phase 4.2.1 — Multi-query fusion (exploratory only)", "",
             "**Four log-derived anchors; NOT scientific retrieval metrics or verified evidence.**", "",
             "| Case | k | Baseline | Focused | Union (pre-budget, theoretical) | Multi (post-budget) | Model visible |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for c in report["comparisons"]:
        lines.append(f"| {c['case_id']} | {c['top_k']} | {c['baseline_anchors']}/{c['total_provisional_anchors']} | "
                     f"{c['focused_anchors']}/{c['total_provisional_anchors']} | "
                     f"{c['union_anchors_pre_budget_theoretical']}/{c['total_provisional_anchors']} | "
                     f"{c['multi_anchors_post_budget']}/{c['total_provisional_anchors']} | "
                     f"{c['multi_anchors_model_visible']}/{c['total_provisional_anchors']} |")
        lines.append("")
        lines.append(f"## {c['case_id']} | k={c['top_k']}")
        lines.append(f"Retrieval calls: baseline={c['retrieval_query_counts']['baseline']}, "
                     f"focused={c['retrieval_query_counts']['focused']}, "
                     f"multi={c['retrieval_query_counts']['multi']}")
        for row in next(r for r in report["per_run"] if r["case_id"] == c["case_id"] and
                        r["top_k"] == c["top_k"] and r["query_style"] == "multi")["facet_diagnostics"]:
            lines.extend(["", f"### {row['step_id']}",
                          f"- Baseline: `{row['baseline_query']}`",
                          f"- Focused: `{row['focused_query']}`",
                          f"- Branch hits: {row['branch_hits']}; union before budget: "
                          f"{row['union_primary_chunks_pre_budget']}; merged after budget: "
                          f"{row['merged_primary_chunks_post_budget']}",
                          f"- First observed anchor rank: {row['first_observed_anchor_rank']}",
                          f"- Provisional anchor IDs lost after fusion: {row['reference_ids_missing_after_fusion']}",
                          f"- Provisional anchor IDs dropped by source register: {row['reference_ids_missing_after_register']}"])
        lines.append("")
    lines.extend(["## Caveats", "", report["interpretation"], "",
                  "Multi-query is opt-in offline preview only; default production policy is baseline.", ""])
    mpath.write_text("\n".join(lines), encoding="utf-8")
    return jpath, mpath

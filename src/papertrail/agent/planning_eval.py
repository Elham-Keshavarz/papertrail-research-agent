"""Offline regression harness for planner classification and dependency validation.

The suite contains hand-written *synthetic examples*; scores are NOT agent answer
quality or externally validated research-plan accuracy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .planning import build_research_plan


VALID_INTENTS = {
    "comparison", "datasets", "sample_size", "metrics", "limitations",
    "methodology", "iterations", "evidence", "definition", "general",
}


class PlanningCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    question: str = Field(min_length=12, max_length=1000)
    expected_intents: list[str] = Field(min_length=1, max_length=5)
    requires_comparison: bool
    paper_id: str | None = None

    @model_validator(mode="after")
    def check_intents(self) -> "PlanningCase":
        if len(set(self.expected_intents)) != len(self.expected_intents) or not set(self.expected_intents) <= VALID_INTENTS:
            raise ValueError("Invalid or repeated expected intent labels")
        return self


class PlanningSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suite_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    annotation_status: str = Field(pattern=r"^synthetic_regression_only$")
    cases: list[PlanningCase] = Field(min_length=4, max_length=100)

    @model_validator(mode="after")
    def unique_cases(self) -> "PlanningSuite":
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Duplicate planning case IDs")
        return self


def evaluate_planning_suite(suite_path: Path) -> dict:
    try:
        suite = PlanningSuite.model_validate_json(suite_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Planning evaluation suite cannot be read or validated") from exc
    per_case = []
    correct_intents = total_predicted = total_expected = exact_matches = comparison_matches = 0
    for case in suite.cases:
        planned = build_research_plan(case.question, paper_id=case.paper_id)
        actual = set(planned.analysis.intents)
        expected = set(case.expected_intents)
        correct_intents += len(actual & expected)
        total_predicted += len(actual)
        total_expected += len(expected)
        exact_matches += int(actual == expected)
        comparison_matches += int(planned.analysis.requires_comparison == case.requires_comparison)
        per_case.append({
            "id": case.id, "expected_intents": sorted(expected), "actual_intents": sorted(actual),
            "exact_match": actual == expected,
            "comparison_correct": planned.analysis.requires_comparison == case.requires_comparison,
            "steps": len(planned.steps), "subquestions": len(planned.analysis.subquestions),
            "dag_validated": planned.validation == "validated_dependency_dag",
            "research_executed": False,
        })
    total = len(per_case)
    return {
        "suite_id": suite.suite_id,
        "annotation_status": suite.annotation_status,
        "cases": total,
        "intent_precision_micro": round(correct_intents / total_predicted, 6),
        "intent_recall_micro": round(correct_intents / total_expected, 6),
        "intent_exact_match_rate": round(exact_matches / total, 6),
        "comparison_accuracy": round(comparison_matches / total, 6),
        "dag_validated_cases": sum(item["dag_validated"] for item in per_case),
        "per_case": per_case,
        "methodology_note": (
            "Synthetic lexical regression labels only; not independently annotated, "
            "not evidence retrieval accuracy, not LLM quality and not research answer correctness."
        ),
        "llm_calls": 0,
        "tool_calls": 0,
    }


def save_planning_report(report: dict, data_dir: Path) -> tuple[Path, Path]:
    suite_id = report["suite_id"]
    if not isinstance(suite_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", suite_id):
        raise ValueError("Invalid suite identifier")
    folder = data_dir / "evaluations"
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / f"{suite_id}_planning.json"
    md_path = folder / f"{suite_id}_planning.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# PaperTrail — Planning Regression ({suite_id})",
        "", "Synthetic examples, **not** external research quality validation.", "",
        f"- Cases: **{report['cases']}**",
        f"- Intent micro-precision: **{report['intent_precision_micro']:.3f}**",
        f"- Intent micro-recall: **{report['intent_recall_micro']:.3f}**",
        f"- Intent exact-match rate: **{report['intent_exact_match_rate']:.3f}**",
        f"- Valid dependency DAGs: **{report['dag_validated_cases']}**", "",
        "| Case | Expected | Predicted | Exact match | Valid DAG |",
        "|---|---|---|---|---|",
    ]
    for item in report["per_case"]:
        lines.append(
            f"| {item['id']} | {', '.join(item['expected_intents'])} | "
            f"{', '.join(item['actual_intents'])} | {item['exact_match']} | {item['dag_validated']} |"
        )
    lines += ["", "## Limitations", "", report["methodology_note"], ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path

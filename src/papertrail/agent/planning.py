"""Phase 3.2: auditable question-aware research planning, with no research execution.

The offline planner is deliberately conservative: it converts question signals to
search tasks, NOT to claims or hypothetical paper contents. Optional model-generated
subquestions remain unverified proposals and must pass strict structural checks.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MAX_SUBQUESTIONS = 5


class PlanningError(ValueError):
    """Invalid plan or an unusable model proposal."""


class QuestionAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intents: list[str] = Field(min_length=1, max_length=MAX_SUBQUESTIONS)
    complexity: Literal["focused", "multi_faceted"]
    requires_comparison: bool
    subquestions: list[str] = Field(min_length=1, max_length=MAX_SUBQUESTIONS)
    ambiguity_flags: list[str] = Field(default_factory=list)
    analysis_method: Literal["rule_based", "llm_proposal_unverified"] = "rule_based"


class ResearchTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    task: str = Field(min_length=6)
    depends_on: list[str] = Field(default_factory=list)
    expected_output: str = Field(min_length=6)
    proposed_tool: Literal["arxiv_search", "pdf_ingestion", "chunk_embedding_index", "hybrid_retrieval", "evidence_review", "cited_synthesis"]
    search_query: str | None = Field(default=None, max_length=500)
    execution: Literal["not_executed"] = "not_executed"


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis: QuestionAnalysis
    steps: list[ResearchTask] = Field(min_length=5, max_length=11)
    validation: Literal["validated_dependency_dag"] = "validated_dependency_dag"

    @model_validator(mode="after")
    def check_dag(self) -> "ResearchPlan":
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError("Plan contains duplicate step IDs")
            if len(set(step.depends_on)) != len(step.depends_on):
                raise ValueError("Plan contains duplicate dependencies")
            if any(dep not in seen for dep in step.depends_on):
                raise ValueError("Plan dependencies must point to earlier tasks")
            seen.add(step.id)
        lookup = {step.id: step for step in self.steps}
        if self.steps[-1].id != "answer" or self.steps[-1].depends_on != ["assess"]:
            raise ValueError("Plan must end with a cited-answer task depending on assessment")
        if lookup["answer"].proposed_tool != "cited_synthesis":
            raise ValueError("Answer step must use the synthesis tool proposal")
        if "ingest" not in lookup or "prepare" not in lookup or "assess" not in lookup:
            raise ValueError("Plan requires ingestion, preparation and assessment")
        if lookup["prepare"].depends_on != ["ingest"]:
            raise ValueError("Preparation must depend on ingestion")
        if "discover" in lookup and lookup["ingest"].depends_on != ["discover"]:
            raise ValueError("Ingestion must depend on discovery when needed")
        if "discover" not in lookup and lookup["ingest"].depends_on:
            raise ValueError("Ingestion must be the first source step for selected papers")
        retrieve = [step for step in self.steps if step.proposed_tool == "hybrid_retrieval"]
        if not retrieve or any(step.depends_on != ["prepare"] or not step.search_query for step in retrieve):
            raise ValueError("Retrieval tasks must have queries and depend on preparation")
        if lookup["assess"].proposed_tool != "evidence_review" or lookup["assess"].depends_on != [step.id for step in retrieve]:
            raise ValueError("Assessment must depend on all retrieval tasks")
        return self


_FACETS: tuple[tuple[str, str, str], ...] = (
    ("comparison", r"\b(compare|comparison|versus|vs\.?|contrast|difference|different|better than|relative to)\b", "Compare the named approaches and their differences using sourced passages"),
    ("datasets", r"\b(datasets?|benchmarks?|corpora|corpus|data sets?)\b", "Identify datasets or benchmarks and how they were used, citing source pages"),
    ("sample_size", r"\b(sample sizes?|how many (examples|samples|instances)|number of (samples|examples))\b", "Identify the reported sample sizes and experimental scope from the source"),
    ("metrics", r"\b(metrics?|f1.?score|accuracy|precision|recall|scores?|measur(e|ed|ement))\b", "Identify reported evaluation metrics and their definitions or context"),
    ("limitations", r"\b(limitations?|weakness(es)?|drawbacks?|failure(s)?|challenges?|risks?|bottlenecks?)\b", "Locate author-reported limitations, failures and qualifying evidence"),
    ("methodology", r"\b(how does|how do|methods?|methodolog(y|ical)|architectur(e|al)|design|pipeline|components?|modules?|approach|implementation|ablation)\b", "Find the implementation, method or component descriptions in source passages"),
    ("iterations", r"\b(iterations?|cycles?|refinement|loop(s)?|maximum number)\b", "Find the retrieval/refinement loop description and any stated limits"),
    ("evidence", r"\b(evidence|sufficien(t|cy)|citation|reliab(le|ility)|faithful|ground(ed|ing))\b", "Find supporting evidence requirements and sufficiency criteria"),
    ("definition", r"\b(what is|what are|defin(e|ition)|mean(s|ing)?|explain)\b", "Find the source's definition and scope of the requested concept"),
)


def _trim_question(value: str, max_length: int = 300) -> str:
    value = " ".join(value.split())
    if len(value) <= max_length:
        return value
    return value[: max_length - 3].rstrip(" ,;:") + "..."


def _facet_queries(question: str, facets: list[str]) -> list[str]:
    # Each query retains the user's original topic; none claims a fact is true.
    q = _trim_question(question, 155).rstrip("?. ")
    descriptions = {item[0]: item[2] for item in _FACETS}
    return [f"{descriptions.get(facet, 'Find evidence relevant to the question')}. Topic: {q}?" for facet in facets]


def analyze_question(question: str, *, paper_id: str | None = None) -> QuestionAnalysis:
    """Reproducible multi-label rule analysis. Never infer answers from keywords."""
    from papertrail.models import ResearchRequest

    normalized = ResearchRequest(question=question).question
    matched = [name for name, pattern, _ in _FACETS if re.search(pattern, normalized, flags=re.IGNORECASE)]
    if "definition" in matched and re.search(r"\bwhat is (the )?(maximum|number|total|value|effect|impact|percentage|result)\b", normalized, re.I):
        matched.remove("definition")
    facets = matched[:MAX_SUBQUESTIONS] or ["general"]
    flags: list[str] = []
    if paper_id is None and re.search(r"\b(this|the)\s+(paper|article|study)\b", normalized, re.I):
        flags.append("paper_reference_without_selected_paper_id")
    if len(matched) > MAX_SUBQUESTIONS:
        flags.append("facet_limit_reached_review_required")
    return QuestionAnalysis(
        intents=facets,
        complexity="multi_faceted" if len(facets) > 1 else "focused",
        requires_comparison="comparison" in facets,
        subquestions=_facet_queries(normalized, facets),
        ambiguity_flags=flags,
    )


def validate_subquestions(raw: object, original_question: str) -> list[str]:
    """LLM output is an untrusted *proposal*, not research evidence or tool commands."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_SUBQUESTIONS:
        raise PlanningError("Expected between one and five proposed subquestions")
    clean: list[str] = []
    for value in raw:
        if not isinstance(value, str):
            raise PlanningError("Proposed subquestions must be strings")
        text = " ".join(value.split())
        if not 18 <= len(text) <= 300 or any(ord(ch) < 32 for ch in text):
            raise PlanningError("Proposed subquestion has invalid size or control characters")
        if re.search(r"(?i)(ignore (previous|all) instructions|system prompt|api[_ -]?key|execute|run command|shell|delete files|https?://|\{\{|<script)", text):
            raise PlanningError("Proposal includes instructions or links rather than research questions")
        if text.casefold() in {existing.casefold() for existing in clean}:
            raise PlanningError("Duplicate proposed subquestion")
        clean.append(text)
    if not original_question.strip():
        raise PlanningError("Missing original research question")
    return clean


def build_research_plan(
    question: str, *, paper_id: str | None = None,
    proposed_subquestions: list[str] | None = None,
) -> ResearchPlan:
    analysis = analyze_question(question, paper_id=paper_id)
    if proposed_subquestions is not None:
        analysis = analysis.model_copy(update={
            "subquestions": validate_subquestions(proposed_subquestions, question),
            "analysis_method": "llm_proposal_unverified",
            "complexity": "multi_faceted" if len(proposed_subquestions) > 1 else "focused",
        })
    steps: list[ResearchTask] = []
    if paper_id is None:
        steps.append(ResearchTask(
            id="discover", task="Identify candidate arXiv papers for the research question",
            expected_output="Candidate paper identifiers with source metadata, not yet verified",
            proposed_tool="arxiv_search", search_query=_trim_question(question, 300),
        ))
    steps.append(ResearchTask(
        id="ingest", task="Locate and extract source PDF passages with page attribution",
        expected_output="Page-attributed text and source provenance pending integrity audit",
        depends_on=["discover"] if paper_id is None else [], proposed_tool="pdf_ingestion",
    ))
    steps.append(ResearchTask(
        id="prepare", task="Prepare page-aware chunks, local embeddings and search index",
        expected_output="Indexed source passages with validated retrieval prerequisites",
        depends_on=["ingest"], proposed_tool="chunk_embedding_index",
    ))
    retrieve_ids: list[str] = []
    for i, subquestion in enumerate(analysis.subquestions, 1):
        step_id = f"retrieve_{i:02d}"
        retrieve_ids.append(step_id)
        steps.append(ResearchTask(
            id=step_id, task=f"Find source passages for research subquestion {i}",
            depends_on=["prepare"], expected_output="Ranked source passages and page citations requiring verification",
            proposed_tool="hybrid_retrieval", search_query=subquestion,
        ))
    steps.extend([
        ResearchTask(
            id="assess", task="Assess evidence coverage, contradictions and provenance before drafting",
            depends_on=retrieve_ids, expected_output="Evidence sufficiency decision with gaps and citation references",
            proposed_tool="evidence_review",
        ),
        ResearchTask(
            id="answer", task="Draft only claims supported by verified citations or abstain",
            depends_on=["assess"], expected_output="Source-cited draft or explicit insufficient-evidence decision",
            proposed_tool="cited_synthesis",
        ),
    ])
    return ResearchPlan(analysis=analysis, steps=steps)

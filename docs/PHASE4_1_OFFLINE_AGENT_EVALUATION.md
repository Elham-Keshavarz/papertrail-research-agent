# Phase 4.1 — Offline Agent Pipeline Diagnostics (v1.5.1)

**Scope:** an offline, reproducible inspection of **what the agent retrieved** versus **what it actually offered the synthesis model**. Also displays an optional *historical* summary of one previously completed Live run. It is **NOT** a human-validated answer-quality benchmark, claim entailment checker, or scientific comparison against other systems.

## Why this exists

In the local FAIR-RAG study, hybrid search with a targeted limitations query exposed chunks on PDF page 18 containing the error taxonomy (e.g., component-level failures and architectural failures). However, `agent-answer` preview showed different page-18 excerpts in the **model-visible** `source_register`, and the first Live answers did not cover the limitations facet. Merely increasing `--top-k` from 3 to 5 did not prove that useful evidence survived the 12-reference source-registration budget.

This phase separates three observable stages:

1. **Retrieval:** does an exact *previously observed* chunk ID occur in the primary hits for its designated subquestion?
2. **Source selection:** did that same chunk ID survive into `synthesis.source_register` and thus into the bounded payload made available to the LLM?
3. **Recorded Live behavior (optional):** what coverage, accepted/rejected claim counts and status did a previous execution *report*? This is a replay of a sanitized summary, not a newly scored answer or a second API call.

## Windows quick start

Run commands from the extracted project's `papertrail-research-agent` directory. Copy `data/` from Phase 3.5.4 **before** running the evaluation. `.env` and its API key are **not needed** for this phase.

```powershell
py -3 -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,search,agent]"
```

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

```powershell
.\.venv\Scripts\python.exe -m papertrail doctor
```

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-evaluate --k-values 3,5 --recorded-live benchmarks/agent_fairrag_observed_live_v1.json
```

**Cost:** zero LLM API calls. Uses the **local cached** PDF, extraction, embeddings and FAISS; embeddings are loaded from the local pinned model cache. External arXiv search/download is disabled (`allow_network=False`), deterministic planner only, and synthesis is preview-only. If local artifacts or model weights are absent, copy/create them first: the evaluation will **not** silently pay for an API call. The computer may still use RAM/CPU and local disk space. The command inherits the Windows early BLAS thread safeguard from v1.4.4.

Output paths (under the configured `PAPERTRAIL_DATA_DIR`, normally `data/`):

- `data/agent_evaluations/fairrag_log_anchors_v1_phase4_1.json`: full structured, bounded diagnostic report.
- `data/agent_evaluations/fairrag_log_anchors_v1_phase4_1.md`: readable per-facet findings.

To run one configuration only:

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-evaluate --k-values 3
```

If a cached artifact is missing, first confirm you copied the whole `data` folder; do not enable network, edit the global model cache or run Live automatically.

## What the labels mean

The bundled `benchmarks/agent_fairrag_log_anchors_v1.json` has **one** multifaceted case, with **four log-derived anchor chunk IDs** across three facets:

- `retrieve_01`: a previously observed passage about standard RAG comparison limitations.
- `retrieve_02`: a previously observed passage listing evaluation metrics.
- `retrieve_03`: two page-18 chunks surfaced in the **separate, targeted** hybrid-search test about FAIR-RAG failure modes. Their absence from the agent's ordinary preview is a diagnostic finding, not proof that the entire question is unanswerable.

These IDs are **not independently checked PDF annotations**. They are selected examples from uploaded local logs. The reference list is intentionally non-exhaustive. Any other returned chunk is **unjudged**, *not* necessarily irrelevant. The labels may not match a different PDF processing layout/model/cache revision; if chunk IDs change, create a new suite rather than treating the old score as comparable.

Reported names such as `observed_anchor_retrieval_fraction` mean: of the specified provisional reference IDs, how many appeared in primary retrieval hits? This is **not standard Recall@k**, because an exhaustive, independently judged qrel set does not exist. `observed_anchor_model_visible_fraction` uses the **same denominator**, now counting only those that reached the evidence register. This is a stage-comparison probe, not a citation faithfulness metric.

The optional `benchmarks/agent_fairrag_observed_live_v1.json` is a **sanitized snapshot** from a prior v1.4.4 run. It records `partial_answer_unverified`, four accepted claims, two rejected claims, two of three facets covered, and one historical LLM call. It contains **no API credential, generated claim text, or provider response**. `llm_calls: 0` for the evaluation means *zero **new** model calls*, not that the historical run was free.

## How to read the report

For each facet and each k, inspect:

- `not_retrieved_ids`: reference snippets **not** returned by that facet's query. Suspect query formulation, ranking, or k limit.
- `retrieved_but_not_selected_ids`: reference snippets retrieved **but dropped** before synthesis. Suspect register capacity / ordering.
- `model_visible_reference_ids`: reference snippets actually available in the bounded evidence register.
- `primary_hits`, `unique_primary_chunks`, `selected_register_entries`: volume and redundancy, **not** quality or confidence.
- `recorded_live_observation`: past reported claim acceptance and subquestion coverage. The evaluator does **not** verify whether those claims are true or entailed by their quoted sources.

The *same* underlying pipeline is executed separately for k=3 and k=5, without LLM requests. No claim that a larger k yields better answers should follow until there is a controlled Live evaluation and human-labeled answer faithfulness review.

## What remains in Phase 4

**Phase 4.2:** build a small human-annotated reference set, with two reviewers when practical; verify PDF page/chunk offsets and adjudicate disagreements. Separate development cases from a holdout set to avoid tuning on the same paper. Define claim-level supported/unsupported/insufficient labels with exact source excerpts.

**Phase 4.3:** replay *saved* model outputs and evaluate claim support, per-facet coverage, contradictions, abstentions, cost/latency and robustness. Human evaluation or an independently audited judge is required before reporting faithfulness. Keep model calls an explicit budgeted opt-in, not part of routine CI.

**Do not** publish the Phase 4.1 fractions as independently verified Recall@k, semantic faithfulness, accuracy, or SOTA performance. Previous Phase 2 page proxy qrels carry the same broad warning.

## If `agent-evaluate` fails before producing a report

The command now reports the **actual failure status**, the failed tool and exception type, and whether the error happened at k=3 or k=5. It does not reveal private paths, model responses or API keys. Do **not** assume the default generic error implies a broken LLM.

From the project root, run the equivalent preview **without** `--allow-network` (zero LLM calls):

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-answer --paper-id 2510.22344 --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --top-k 3 --max-refinement-rounds 0 --mode preview
```

Inspect `status`, `errors`, `tool_events`, and `trace`. Run the same command with `--top-k 5` only when k=3 succeeds. If the status is `offline_source_missing`, verify the copied `data/papers`, `data/extracted` and `data/chunks` directories; if `tool_failed`, note the failed tool and `error_type`. If a `citation_validation_failed` occurs, inspect the source register logic. Do not run paid Live as a troubleshooting step.

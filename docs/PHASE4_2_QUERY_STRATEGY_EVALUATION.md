# Phase 4.2 — Paired Query Strategy Diagnostics (v1.5.2)

**Scope:** one **offline**, controlled A/B comparison of the production `baseline` search queries and an *opt-in, deterministic* `focused` query-rewriting candidate. Designed to diagnose missing passages, **not** to deploy a trained query policy or to make validated scientific quality claims.

## Why

The Phase 4.1 real Windows run found 2/4 provisional anchor chunk IDs in the main Agent query pipeline at both k=3 and k=5. Neither of two independently observed **log-derived** chunks discussing failures on page 18 appeared in `retrieve_03`. A separately written, targeted limitations query had found them. This suggests query formulation/ranking might matter. It does **not** prove which new query will perform best across papers.

## Windows installation from scratch

Extract the full Phase 4.2 ZIP and open its **inner** `papertrail-research-agent` folder in VS Code. If your download folders are siblings, copy the cached data from v1.5.1:

```powershell
Copy-Item -Path "..\..\papertrail_phase4_1_1_offline_evaluation_error_diagnostics_full\papertrail-research-agent\data" -Destination ".\data" -Recurse -Force
```

Confirm you are in the right directory and the source data exists:

```powershell
Get-Location
Test-Path .\pyproject.toml
Test-Path .\data
```

All three `Test-Path` checks for files/data should return `True` when appropriate. Then:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,search,agent]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m papertrail doctor
```

**Main Phase 4.2 A/B test — no network, no API key, no LLM calls:**

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-query-evaluate --k-values 3,5
```

Outputs:

- `data/agent_evaluations/fairrag_log_anchors_v1_phase4_2_query_comparison.json`
- `data/agent_evaluations/fairrag_log_anchors_v1_phase4_2_query_comparison.md`

Optional: inspect the focused query plan **without** an LLM call, answer or internet:

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-answer --paper-id 2510.22344 --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --top-k 3 --max-refinement-rounds 0 --query-style focused --mode preview
```

Use `--query-style baseline` or omit the flag to get the unchanged default. Focused is explicitly **rejected in Live**, to keep the experimental policy from silently affecting cost-bearing answers. No `.env` or API key is necessary; transferring `.env` is unnecessary for this phase. Installing dependencies initially requires internet, but the benchmark explicitly prevents paper-network retrieval/model API calls and uses local cached model weights.

## What is measured

Each pair uses the same question, paper ID, per-facet query slots, `top-k`, local embedding model, FAISS/BM25 RRF, source register limit, and cache. **Only the per-facet search string changes.** This is an experimental intervention, not a new truth label.

Per facet:

- `not_retrieved_ids` — observed anchor IDs absent from primary hits
- `retrieved_but_not_selected_ids` — retrieved but not in the model-visible register
- `first_observed_anchor_rank` — first primary rank among supplied observed anchor IDs, or null
- `observed_first_anchor_reciprocal_rank` — 1/rank on those examples, 0 if absent
- `query` — actual baseline or focused search string sent to retrieval

Per case and k: `observed_anchor_retrieval_fraction`, `observed_anchor_model_visible_fraction`, `observed_facet_anchor_mrr` and paired deltas. **These are descriptive statistics on a handful of observed anchors, NOT gold-standard Recall@K or MRR**. Outside these examples, retrieved chunks are **unjudged** rather than irrelevant. A targeted query written after seeing the failed case risks evaluation leakage, so a positive delta is a hypothesis, not proof of generalization.

## Human review protocol (not yet performed)

`benchmarks/phase4_2_manual_review_queue_v1.json` is a **pending** two-reviewer worksheet, *not* verified ground truth. To create real qrels:

1. Open the exact downloaded PDF and verify its SHA-256 against the cached extraction and chunk manifest, including 1-based PDF pages and reading-order issues.
2. Independently read each page and surrounding sentences. Reviewers A/B must check the actual query facet, the exact excerpt, factual meaning and whether the chunk alone is sufficient. Mark supported / unsupported / insufficient; record reviewer notes and exact text spans. Do **not** judge using the retrieved rank alone.
3. Reconcile disagreements, record adjudicator and date. Separate a held-out group of other papers **before** selecting the query policy. Four anchors from the same paper are **not** an independent held-out study.
4. Only after agreement, build a versioned, externally audited qrel set that includes every relevant primary chunk possible and evaluate conventional Recall@K, MRR and query robustness. Keep these new labels separate from old provisional logs.
5. Before claiming scientific QA quality, also run **claim-level semantic entailment / contradiction review** (planned Phase 4.3). Exact string presence does not imply an accurate claim.

No reviewer labels are filled or claimed to be verified in this release. There is no automatic strategy promotion. Continue to use baseline for Live until the multi-paper holdout supports changing it.

## Troubleshooting

If `agent-query-evaluate` fails, confirm `Test-Path .\data` is true and run the standard baseline preview. If the model cache is missing, restore the local cache; do not turn on paid Live to debug. The program reports which `case`, `top_k`, `query_style` and failed tool if preview fails. Share only the terminal output and the generated Markdown report; **never upload `.env`**.

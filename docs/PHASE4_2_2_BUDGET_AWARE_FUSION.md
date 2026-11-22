# Phase 4.2.2 — Budget-Aware Fusion (experimental, offline)

## Why this was built

In the user's actual Windows Phase 4.2.1 results for FAIR-RAG arXiv `2510.22344`, baseline retrieved **2/4** provisional observed anchor IDs, focused **3/4**, and RRF multi-query **3/4**, with **4/4** in the theoretical pre-budget union. The RRF implementation discarded an observed limitations anchor in its final fixed-budget shortlist. These are *non-exhaustive, previously observed* chunks, **not validated ground-truth relevance labels**. Other retrieved chunks have not been judged.

This version introduces a controlled *budget-aware* alternative to cross-query RRF. Its purpose is to assess whether fairer representation of both local retrieval branches preserves more useful, distinct passages **under the identical top-k output limit**, not to declare any approach better before actual testing.

## Two algorithms, both with the same limits

- `multi` (existing): independently validate baseline and focused results; exact-ID deduplication; cross-query reciprocal rank fusion, `k=60`; choose highest-ranked `top-k` per facet.
- `budgeted` (new): independently validate the SAME two branches and enforce the same `top-k` limit; select distinct items by **focused-first, baseline-second rank-list interleaving**, filling available positions without duplicates. Starting with the focused branch is a fixed, declared experimental diversification policy, **not** conditioned on the anchor IDs or their judgment. The code does not perform semantic reranking, content relevance checking, or evidence verification. Each displayed fused result still carries the RRF score as a ranking diagnostic, not a confidence value.

For either method, source collisions with mismatched page/citation/text/context **fail closed**. Both make twice as many local retrieval calls as baseline. Budgeted can perform worse than either baseline or RRF for other questions, and no claim of generalization is made.

**Scope:** `--query-style budgeted` is allowed only for local deterministic `--mode preview`, just like focused and multi. Live LLM synthesis, network and LLM-planner calls are blocked in the CLI and core API. The production/default style remains `baseline`.

## Windows PowerShell — fresh full ZIP

Extract the `v1.5.4` full ZIP and open its **inner** `papertrail-research-agent` directory in VS Code. If your terminal is in the outer folder, first run `cd .\papertrail-research-agent`. Copy data only, assuming version folders are adjacent in `Downloads`:

```powershell
Copy-Item -Path "..\..\papertrail_phase4_2_1_multi_query_retrieval_full_v1_5_3\papertrail-research-agent\data" -Destination ".\data" -Recurse -Force
Test-Path .\pyproject.toml
Test-Path .\data
```

Both `Test-Path` calls must return `True`. Do not copy `.venv` or `.env` (not needed for this benchmark). Build a fresh environment, then test:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,search,agent]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m papertrail doctor
```

Run the offline four-way comparison at two fixed budgets:

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-budget-evaluate --k-values 3,5
```

This performs four full Preview configurations (baseline / focused / multi-RRF / budgeted) at each of 2 `top-k` settings, or **8 workflows** total. For three retrieval facets, the per-configuration query budgets are `3 / 3 / 6 / 6` calls. The actual benchmark uses locally cached model weights and source artifacts, but makes **0 LLM API calls and no arXiv network requests**.

To inspect just one Budgeted result, without running the benchmark:

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-answer --paper-id 2510.22344 --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --top-k 3 --max-refinement-rounds 0 --query-style budgeted --mode preview
```

## Files and fields

Reports are written to `data\agent_evaluations\`:

- `fairrag_log_anchors_v1_phase4_2_2_budget_fusion.json` — all four per-run diagnostics, selected references and lost-ID comparison.
- `fairrag_log_anchors_v1_phase4_2_2_budget_fusion.md` — one-page A/B plus per-facet Budgeted evidence details.

For PowerShell UTF-8:

```powershell
Get-Content -Encoding UTF8 .\data\agent_evaluations\fairrag_log_anchors_v1_phase4_2_2_budget_fusion.md
```

`union_anchors_pre_budget_theoretical` is the hypothetical union before any selection cap; it is not the final output. Compare `rrf_anchors_post_budget` against `budgeted_anchors_post_budget`, then check `budgeted_anchors_model_visible`. The `lost_to_fusion` and `lost_to_source_register` fields distinguish ranking-budget loss from later register loss. A higher number **does not** demonstrate scientific recall, answer faithfulness, or broad usefulness.

## Interpretation and exit criteria

These four IDs were sourced from *previous logs* for a **single paper** and may overfit the chosen query interventions. No source-PDF read-order audit, two-reviewer completed adjudication, held-out multi-paper study, or semantic entailment validation has been performed. The file `benchmarks/phase4_2_manual_review_queue_v1.json` still contains unreviewed examples; it must not be used as verified qrels.

**Stop rule to control scope:** After comparing the two fusion methods on Windows, do not add endless hand-tuned FAIR-RAG query variants. Record the outcome (including if budgeted ties or loses), keep `baseline` as the default, and move to a small manually reviewed multi-paper sample and Phase 4.3 (answer faithfulness), followed by Streamlit and GitHub publishing.

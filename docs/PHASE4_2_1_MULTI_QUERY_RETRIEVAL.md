# Phase 4.2.1 — Multi-Query Retrieval (provisional, offline)

## Motivation and evidence limits

On the user's Windows run of v1.5.2, the **baseline** search found 2/4 *previously observed* provisional anchor chunk IDs; **focused** found 3/4. Focused recovered 2/2 observed limitations anchors but lost the one comparison anchor. The **union** of two runs contains 4/4, but this is an **unbounded, theoretical pre-fusion upper bound**, not a tested merged-agent output or a validated recall score.

This phase compares baseline, focused and multi within the SAME paper and top-k. All four chunk IDs were chosen from previous logs. They are NOT independent human-adjudicated qrels; unrelated chunks are **unjudged** and cannot be called false positives. We cannot claim scientific Recall@K, generalization, answer faithfulness, hallucination reduction, or a production performance win.

## What actually runs

The `multi` query style is a **preview-only** experimental option in the original LangGraph tool workflow. For each planned retrieval facet:

1. Preserve the baseline DAG, source selection and baseline query.
2. Generate a deterministic focused alternative from the question and its facet (no PDF labels or paper-specific answers used).
3. Run two **real local hybrid-search tool calls**, validating page, citation and chunk structure **independently**.
4. Deduplicate by exact chunk ID; if the same ID has conflicting text, page, citation or context, **fail closed**.
5. Rank the union by unweighted *cross-query reciprocal rank fusion* (RRF, k=60). Ties are deterministic: best source rank, baseline before focused, then chunk ID. RRF is NOT confidence or factual correctness.
6. Retain **at most `top-k` chunks per facet**, preserving the existing source-register selection budget (up to four per facet, subject to global limits). No anchor-specific selection, manual tweaking, or labels enter the fusion logic.
7. Audit the per-branch search strings and returned chunk IDs, fused hits, source register and tool-call counts. If any branch fails, the whole facet fails; do not silently fall back.

Compared to baseline, multi makes **2x retrieval calls** (six for three facets) for each paper/k setting; `max_tool_calls` remains bounded. Cached embedding/index resources may still be expensive in CPU/RAM; no claim of lower latency is made. `max_refinement_rounds=0` makes every ablation equally reproducible.

**Guardrails:** `--query-style multi` rejects Live generation, LLM planning and arXiv/network opt-in; it does not alter default baseline nor run a model. It does not invent supporting quotations or turn unreviewed evidence into fact.

## Windows PowerShell — full clean start

Extract the full ZIP and open the inner `papertrail-research-agent` folder. If PowerShell is in the ZIP's outer directory, run `cd .\papertrail-research-agent` first.

Copy **only data**, not `.venv` or `.env`, from the v1.5.2 folder (both extract folders must be adjacent in Downloads):

```powershell
Copy-Item -Path "..\..\papertrail_phase4_2_query_strategy_ab_offline_full_v1_5_2\papertrail-research-agent\data" -Destination ".\data" -Recurse -Force
```

```powershell
Test-Path .\pyproject.toml
Test-Path .\data
```

Both should be `True`. Install dependencies, test, and check health:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,search,agent]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m papertrail doctor
```

Run the full OFFLINE, no-API benchmark:

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-multi-query-evaluate --k-values 3,5
```

Alternatively inspect one multi-Query run without LLM (does not produce an answer):

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-answer --paper-id 2510.22344 --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --top-k 3 --max-refinement-rounds 0 --mode preview --query-style multi
```

Reports (new files) are created in `data\agent_evaluations`:

- `fairrag_log_anchors_v1_phase4_2_1_multi_query.json` — full three-way per-run records, per-facet query branch traces, lost-at-fusion and lost-at-register IDs, tool-call budget.
- `fairrag_log_anchors_v1_phase4_2_1_multi_query.md` — concise A/B/Multi side-by-side, with theoretical pre-budget union **explicitly separate** from observed post-budget results.

For correct UTF-8 PowerShell rendering:

```powershell
Get-Content -Encoding UTF8 .\data\agent_evaluations\fairrag_log_anchors_v1_phase4_2_1_multi_query.md
```

No `OPENAI_API_KEY` is required or used. The first installation of dependencies or the embedding model may require internet, but the benchmark itself requires **local cached PDF/extracted data/chunks/embeddings** and **does not opt into any network access**.

## How to interpret results

If `union_anchors_pre_budget_theoretical=4` but `multi_anchors_post_budget=3`, the selected ranking/`top-k` budget discarded an observed anchor, **not** the retriever: both query results contained the chunk. Investigate `lost_to_fusion` and per-facet `reference_ids_missing_after_fusion`. If post-budget contains a reference but `multi_anchors_model_visible` excludes it, source-register selection is the bottleneck.

No metric should be labeled ground-truth Recall or definitive improvements until (a) the source PDF is independently reviewed, (b) multiple-paper held-out cases are annotated, and (c) semantic entailment and contradiction checks are performed. The existing `benchmarks/phase4_2_manual_review_queue_v1.json` remains **unadjudicated** and is not a scoring file.

## Safety and troubleshooting

- `status: ready_for_llm_synthesis` means structurally adequate evidence for the *preview*, not verified accuracy.
- If the command prints `tool_failed`, check `tool_events` from preview. PDF/source validation failures stop execution; no fabricated answer is returned.
- If the command fails due to `tool_budget_exhausted`, cache the data first and stay at the default `max_tool_calls=12`; the per-paper query budget is *not silently expanded*.
- If Windows reports `WinError 112`, free disk space and create a fresh `.venv` before testing. Keep `data/` untouched.
- If `OpenBLAS` memory allocations fail, keep the `cli_bootstrap.py` default thread cap; do not debug by making paid API calls.
- If the real model weights aren't present in the local Hugging Face cache, restore those weights from the previous setup; do not switch `--allow-network` on in experimental multi mode.

## Next

This is a controlled engineering diagnosis, **not** a held-out study. The next work should prioritize 2–3 genuinely new papers, independent annotation of query-to-passage relevance, retrieval Recall@K/nDCG/MRR with frozen labels and eventual answer entailment audits (Phase 4.3). No deployment policy changes until those checks pass.

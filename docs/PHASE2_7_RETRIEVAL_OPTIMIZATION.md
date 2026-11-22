# Phase 2.7 — Held-out Retrieval Optimization & Label Audit (v0.9.0)

## What changed

The prior **eight-question provisional** FAIR-RAG page-label evaluation gave:

| Metric | BM25 | FAISS | Unweighted Hybrid |
|---|---:|---:|---:|
| MRR@10 | 0.895833 | 0.833333 | 0.733333 |
| Precision@1 | 0.875 | 0.750 | 0.625 |
| Recall@10 | 1.000 | 1.000 | 1.000 |

These figures came **from the user's v0.8.0 Windows run**, not from this release. They are NOT independently reviewed passage-level metrics. They show why automatically preferring RRF is not justified.

New code in this release:

- Configurable **weighted RRF**: `--dense-weight`, `--bm25-weight`, values in 0..10 (not both zero). We **never add raw cosine to BM25**. Default `(1,1)` reproduces previous scores and ranking.
- Optional **per-page diversity**: `--max-hits-per-page 0` (default/off) or `1..5`, applied after fusion. We do not change chunk text, provenance or RRF score. Capping can return **fewer** than K if insufficient pages/candidates exist; interpret P@K accordingly.
- Frozen **14-candidate grid**: seven transparent dense/BM25 weight pairs × page cap `{0,2}`.
- Deterministic query-level train/holdout split with a recorded seed, fixed grid, and explicit training/validation metrics. Six questions train and two validate for the bundled eight-question suite. Weight selection uses **training labels only**; validation numbers are computed *after* selection and are diagnostic, **not** used for selection.
- **Candidate caching across grid weights**: one FAISS request and one BM25 query per question. No model calls per weight combination and no API key.
- Possible missing page labels are listed for **manual review**: an unlabeled page in top-3 is flagged, never implicitly labeled relevant or irrelevant.
- Reproducibility metadata: evaluation-suite SHA-256, corpus-chunk SHA-256, model/revision, split seed, case IDs, candidate scores and selected grid parameters.
- The optimization **does not auto-deploy** any new weight or page cap and writes files with a new suffix. Existing saved PDFs, chunks, vectors and FAISS indexes are not modified.

## Windows PowerShell (one line per command)

Work from the `papertrail-research-agent` directory after copying the **entire** previous `data` folder:

```powershell
py -3 -m venv .venv
```
```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,search]"
```
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
```powershell
.\.venv\Scripts\python.exe -m papertrail doctor
```
```powershell
.\.venv\Scripts\python.exe -m papertrail optimize-retrieval --paper-id 2510.22344 --suite benchmarks/fair_rag_pages_v1.json --metric mrr_at_k --k 5 --seed 42
```

The last command writes:

- `data/evaluations/2510.22344_fair_rag_pages_v1_rrf_optimization.json`
- `data/evaluations/2510.22344_fair_rag_pages_v1_rrf_optimization.md`

**Read the report**, including the `flagged_unlabeled_pages` table, and check candidate page labels against the original article before interpreting performance. A 2-question holdout is far too small to establish generalization, especially with **provisional page labels**.

## Non-deploying manual examples

The default remains **unchanged**, exactly as Phase 2.5:

```powershell
.\.venv\Scripts\python.exe -m papertrail hybrid-search --paper-id 2510.22344 --query "Which datasets were used to evaluate FAIR-RAG in the experiments?" --top-k 5 --context-window 1
```

Try alternative weights and a maximum of two primary chunks per page **only when explicitly requested**:

```powershell
.\.venv\Scripts\python.exe -m papertrail hybrid-search --paper-id 2510.22344 --query "Which datasets were used to evaluate FAIR-RAG in the experiments?" --top-k 5 --context-window 1 --dense-weight 1 --bm25-weight 2 --max-hits-per-page 2
```

To evaluate a manually chosen setting, use:

```powershell
.\.venv\Scripts\python.exe -m papertrail evaluate-retrieval --paper-id 2510.22344 --suite benchmarks/fair_rag_pages_v1.json --k-values 1,3,5,10 --dense-weight 1 --bm25-weight 2 --max-hits-per-page 2
```

**Caution:** the last command overwrites the previous Phase 2.6 report with the same stem, but the file records which weights/cap were used. Copy the default report first if you need a side-by-side record. The optimization command above writes its *own* distinct report and does not overwrite Phase 2.6 output.

## Ground-truth review process

1. Open `https://arxiv.org/pdf/2510.22344`, and compare raw PDF page locations against the extracted JSON and the `flagged_unlabeled_pages` candidates; **do not mark a page relevant just because the retriever returned it**.
2. Examine especially `ablation_components`: preliminary labels include page 14, but the previous user-visible output showed a section heading and analysis on page 13. Check **whether page 13 actually supports the query** before adding it.
3. Confirm the 30% reduction in extracted characters (178,156 → 125,031), all-page two-column detection, and tables/figures. This release does **not** silently claim those issues solved.
4. Improve benchmark independence: expand to more papers and independently reviewed passage spans (not just page-proxy judgments) *before* treating optimization scores as research findings.
5. Leave the production defaults unchanged until a **separate, reviewed, larger test corpus** supports deployment.

## Methodology caveats

- These are source-page proxy labels, not directly assessed relevance of individual chunk text, and not correctness of generated answers.
- All hyperparameters in the grid are tuned only on a six-question training subset. The two-query holdout still comes from the same paper and shares the same annotation uncertainties.
- Page diversity may artificially inflate page-level recall by spreading candidates over pages; it can lower precision or suppress useful repeated evidence. It is explicitly disabled by default.
- Training a weight on this tiny set can overfit. The report uses `deployment_recommendation` strings that **never** imply automatic adoption.
- Rank fusion `score` is not a probability and is not comparable with cosine scores or BM25 scores.
- This milestone remains local/offline after model installation. Phase 3 (agentic planning/grounded synthesis) is **not** implemented here.

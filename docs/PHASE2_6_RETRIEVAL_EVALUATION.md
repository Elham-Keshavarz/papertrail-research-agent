# Phase 2.6 — Retrieval Evaluation (v0.8.0)

## Goal

Compare **BM25**, **FAISS** and **hybrid RRF** on *the same query set, existing article and chunk files*. This is retrieval-only evaluation. PaperTrail still does **not** answer research questions or run an autonomous agent.

New files:

- `benchmarks/fair_rag_pages_v1.json` — versioned, **provisional** eight-question page-relevance suite for arXiv `2510.22344` (labels drafted from previously observed user retrieval logs, not independently adjudicated).
- `src/papertrail/retrieval/quality_eval.py` — typed input validation, offline scoring, three-method orchestration, deterministic reports.
- `tests/test_quality_eval.py` and `tests/test_quality_eval_cli.py` — synthetic offline regression tests; no network/API/real model necessary.
- `data/evaluations/<paper-id>_<suite-id>.json` and `.md` — generated only when CLI runs; ignored by Git.

## Run (from repository root)

The source PDF is **not** included in source ZIPs. Copy the previous `data/` directory intact before evaluating. The last verified corpus had 421 chunks with the pinned 384-dimensional MiniLM embedding model. Re-embedding is unnecessary if all integrity checks pass.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m papertrail doctor
.\.venv\Scripts\python.exe -m papertrail evaluate-retrieval --paper-id 2510.22344 --suite benchmarks/fair_rag_pages_v1.json --k-values 1,3,5,10
```

The final command uses only local embeddings and the FAISS index; **no LLM API key** or remote inference is required. Its first run may load the pinned MiniLM model from Hugging Face cache; the model instance is reused across test questions. If your model was not previously downloaded, initial model retrieval needs internet.

CLI stdout is a compact JSON summary of macro averages. The full ranked page traces and per-question results are written to `data/evaluations/2510.22344_fair_rag_pages_v1.json` and `.md`. Do **not** treat an unexecuted benchmark as a measured result.

## Labels and relevance unit

The evaluation suite represents a question by its text and one or more **relevant PDF page numbers**. Page identifiers are 1-based and correspond to `#page=N` citations in previously retrieved evidence. User-supplied retrieval excerpts guided provisional choices; pages have **not** been exhaustively adjudicated by two independent reviewers. To use the evaluation in a public portfolio, open the original PDF and manually check every labeled page, add missing relevant pages, record the review protocol, and only then update `annotation_status` to `reviewed`.

A **retrieved primary chunk** is considered a proxy relevant hit whenever its page belongs to the question's labeled page set. This is a deliberately coarse proxy: an off-topic passage on a relevant page can be counted as relevant, and an actual supporting passage on an unlabelled page can be counted as irrelevant. It is **not** claim support, answer correctness or graded passage relevance.

The suite does not use expected answers or label text as retrieval queries. The same plain user question is sent independently to every method, with identical K values and matching document and embedding revision.

## Metrics (page-proxy, top K *primary chunks*)

Let `R` be the fixed set of labeled relevant pages for one question; `H_K` be the ordered first K **primary chunks** returned by a method, without neighboring-context expansion; `page(h)` be each hit's PDF page. K is the requested rank cutoff, even if a sparse method returns fewer hits.

- **Precision@K** = `count(h in H_K such that page(h) in R) / K`. Multiple hits from the same relevant page count separately in precision because the retrieved units are chunks.
- **Recall@K** = `count(unique page(h) in R across H_K) / len(R)`. Duplicate page hits do not raise recall.
- **MRR@K** = `1 / rank_of_first_relevant_primary_chunk` when any labeled page appears among the first K results, otherwise 0. This is a one-query reciprocal rank, not an LLM confidence score.
- **Hit@K** = 1 when at least one labeled page appears among the first K results, otherwise 0.

Results are **macro-averaged** over queries, weighting every question equally. Full counts and cited page rankings are retained for auditing. For this phase, hybrid neighbor context is deliberately **excluded** from the scores so that results have a common primary-chunk unit. Phase 2.5's `evaluate-benchmarks` still provides a separate name-coverage diagnostic **including context**.

These precise metrics differ from canonical passage-level retrieval metrics when judgments are for entire pages rather than individual chunks. Label them `page-proxy` in GitHub/README/portfolio text. The page labels may be incomplete, and eight questions from one paper cannot demonstrate generalization.

## Data integrity / reproducibility

- CLI verifies the cached chunk file, normalized embeddings and their SHA-256 hashes before search.
- Every returned chunk identifier and page attribution is checked against the verified chunk list.
- A suite for a different paper is refused; missing labeled pages fail rather than silently producing zero recall.
- Each saved report records suite ID, provisional/reviewed status, model name and pinned revision, vector count, chunk SHA-256, all K values, per-question ranked pages, macro averages and limitations.
- The same encoder instance is reused for all eight questions; it does not cache query vectors or leak labels into the search.
- A synthetic test fixture validates no-hit handling, duplicate page recall, strict K parsing, MRR, macro averaging, report serialization and error paths.
- Results are deterministic for an unchanged corpus, software environment and local model, up to normal numerical runtime differences.

## Interpretation and limitations

The previously measured **75% FAISS vs 100% hybrid** was the **four-name coverage diagnostic for one question**. It must **not** be reused as a Recall@K or MRR result for this new benchmark. The new macro metrics only become available after running `evaluate-retrieval` on your Windows machine. Higher Recall does not necessarily mean higher Precision, and BM25 may outperform RRF on some queries.

PDF layout processing lost approximately 30% of the prior character count (178,156 down to 125,031), and all 30 pages were labeled two-column by the heuristic; this change requires an independent completeness/reading-order audit. Table and equation fidelity remain unresolved. The report deliberately has `pdf_extraction_quality_verified: false`.

For more defensible future evaluation, manually annotate exact relevant passage spans/chunk IDs and score multi-document, held-out queries. Preserve both successes and failures in the GitHub report. Do not train/modify retrievers against the test suite without keeping a separate held-out test split.

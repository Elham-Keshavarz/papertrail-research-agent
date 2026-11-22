## v1.6.2 — Streamlit workflow progress (GitHub release candidate)

- Added real workflow milestone progress in Streamlit Research: completed LangGraph nodes, per-facet retrieval counts, evidence assessment, Preview/Live answer stage and finalization. The progress bar is updated synchronously as LangGraph streams actual state; the original CLI `.invoke` path remains unchanged.
- Added import progress for source ingestion, chunking, local embeddings, FAISS indexing and readiness validation. No additional model/API calls are made by progress reporting.
- Stage percentages represent completed milestones, **not elapsed time, PDF bytes, model-weight loading counts, or scientific confidence**. Long embedding/model-loading steps may hold the same percentage; on exceptions the bar does not falsely complete.
- Extended stub UI tests, service callback tests and graph stream/progress regression tests. Native LangGraph + Streamlit browser smoke check is still required on Windows.
- Maintained Preview's offline behavior and explicit Live/network consent. This is still a local release candidate, not a published GitHub release.

## v1.6.1 — Phase 6 GitHub release candidate (not yet published)

- Refreshed the README landing page with truthful optional Preview/Live behavior, Windows commands, citations and evaluation limitations.
- Added GitHub checklist, community standards, Security and Contributing files, and a local release/secret-pattern scanner with regression tests.
- Changed deprecated Streamlit `use_container_width` calls to `width="stretch"` and added a regression test.
- Extended Git ignore rules and CI checks. The baseline retrieval policy and optional Live request controls are unchanged.
- Version is a **release candidate**: it has not been pushed and needs a final Windows UI smoke check.

## 1.6.0 — Phase 5 Streamlit UI

- Added Streamlit app with actual LangGraph Preview/Live integration; Live requires key plus explicit user approval.
- Added local paper selection, optional arXiv discovery/import with explicit network consent, and visual retrieved evidence/citations.
- Added secure local Human Review queue authoring and decision capture, preserving non-entailment limitations.
- Kept experimental query styles Preview-only; core retrieval and answer-validation defaults unchanged.
- Added UI service unit tests and UI documentation. Streamlit-native browser execution remains to be verified on Windows.

## v1.5.5 — Phase 4.3: Faithfulness guard and human review (offline)

- Re-check source ID, facet, page, citation, literal quote and numeric anchoring after successful Live synthesis validation; attach an explicit non-semantic `faithfulness_audit` (no extra model call).
- Distinguish verbatim claim extraction from quote-anchored paraphrase while marking **both** `semantic_support: not_evaluated` and `human_review_required: true`.
- Add 10 synthetic guard-regression scenarios, including negation inversion deliberately *not* semantically certified.
- Add CLI commands for local bundle auditing, private manual review queue export and summarizing user-entered judgments; no LLM/FAISS/arXiv access.
- Maintain baseline as default and existing citation/source-register budgets. Real PDF fidelity and semantic entailment still require independent human validation.

## v1.5.4 — Phase 4.2.2 offline Budget-Aware Fusion A/B

- Added an opt-in `budgeted` multi-query path that **interleaves focused/baseline ranked lists** under the existing per-facet `top-k` limit; retains strict citation and payload checks, no LLM/network use, and unchanged default baseline policy.
- Added `agent-budget-evaluate` for an eight-run, four-policy paired comparison at `top-k=3,5`, showing actual post-budget hits, model-visible anchors, hypothetical pre-budget union, lost-at-fusion IDs, and doubled local retrieval calls.
- Fixed a prior multi-query diagnostic bug where loop-variable shadowing could mislabel an RRF (`multi`) run as `focused` in detailed report records. Added regression tests for report generation and style identity.
- Added deterministic merger tests, evaluation tests, and optional native LangGraph integration coverage. This remains **one unverified paper and four log-derived anchors**, not publication-quality qrels, verified semantic correctness, or evidence of generalizable improvement.

## v1.5.3 — Phase 4.2.1 offline multi-query retrieval (experimental)

- Added `--query-style multi` opt-in deterministic preview routing via the original LangGraph tools: baseline plus focused retrieval per facet, independently validated, source provenance-checked, deduplicated and RRF fused under the original per-facet top-k budget.
- Added `agent-multi-query-evaluate` three-way real retrieval comparison across top-k 3/5. Reports **theoretical pre-budget union** separately from actual fused post-budget/model-visible anchors, with per-facet lost-anchor diagnostics and the doubled local tool-call cost.
- Preserved baseline defaults and fail-closed guardrails: no paid Live, LLM planner, arXiv networking, content invention, or independent entailment claims in this experimental mode.
- Synthetic, shim-graph and optional native LangGraph integration tests added. Multi-paper independent human annotation is still outstanding; log-derived labels remain provisional and non-exhaustive.

## v1.5.2 — Phase 4.2 Experimental Query A/B (offline only)

- Adds `agent-query-evaluate`: same-paper/same-k paired `baseline` versus deterministic `focused` query interventions using the existing LangGraph hybrid retrieval and selection pipeline. No API key or new LLM calls.
- Reports actual per-facet query strings, observed anchor ranks, provisional anchor coverage and selected-source survival. Emits JSON/Markdown without treating log-derived examples as independently judged qrels.
- Adds `agent-answer --query-style focused --mode preview` for optional inspection; rejects experimental query mode in paid Live. Default remains `baseline`.
- Adds a **pending, unverified** two-reviewer PDF annotation queue, a source-audit protocol and native/simulated pipeline tests. No automatic deployment or factual entailment certification.

## v1.5.1 — Phase 4.1 offline preview failure diagnostics

- Preserve fail-closed evaluation but display sanitized run status, failed tool, error type, last graph node and top-k on failed offline previews.
- Add regression tests ensuring raw provider/tool text and secrets never appear in diagnostics.
- No new model calls, network requests, or changes to evidence scoring or citations.

## v1.5.0 — Phase 4.1: offline, provisional agent pipeline evaluation

- Added `agent-evaluate` (offline, deterministic, no LLM API, no arXiv network) to run the existing LangGraph preview with multiple top-k values and inspect primary retrieved chunk IDs versus the bounded model-visible source register.
- Bundled **one** log-derived FAIR-RAG multifaceted diagnostic case. Its 4 source chunk IDs are provisional observations from local logs, **not** independently annotated ground truth or exhaustive relevance judgments.
- Built optional historical Live replay from the already-shared v1.4.4 output: 4 accepted claims, 2 rejected, 2/3 facets covered; **no new LLM call**. This is a historical observation, not verified claim faithfulness.
- Added bounded suite/snapshot validation, fail-closed offline guards, per-facet missing-vs-dropped evidence diagnostics, JSON/Markdown reports, and regression tests.
- All previous phases and quote selection, BLAS startup guard, strict citation validation and opt-in Live mode remain intact.
- Future Phase 4 work: independently adjudicated labels, multi-paper holdout, human claim entailment, contradictions, measured cost/latency; do not claim current quality percentages are scientifically validated.

## v1.4.4 — Windows OpenBLAS startup resource guard (Phase 3.5.4)

- Real Windows v1.4.3 execution ran all tests but failed before both `agent-answer --mode preview` and `--mode live` with repeated OpenBLAS memory allocation errors. No evidence of an answer-model API call in this log.
- At CLI startup (not plain library import), before NumPy/FAISS imports, limit OPENBLAS/OMP/MKL/NUMEXPR worker counts to 1 by default; allow explicit `PAPERTRAIL_MATH_THREADS=1..8` or opt-out `PAPERTRAIL_LIMIT_MATH_THREADS=0` in the launching shell. Avoid including API credentials in diagnostics.
- Added subprocess boot tests for inherited thread counts, customized caps, opt-out and invalid values. This is a resource fix, not an answer-quality or citation-fidelity claim.
- `--mode preview` remains the free/no-key first check; DO NOT run a paid Live call if Preview is failing. Insufficient system RAM/pagefile can still make OpenBLAS fail even with one thread.

## v1.4.3 — Phase 3.5.3: deterministic quote-anchor selection

- Addressed the first real live result (`partial_answer_unverified`): one of six proposed claims passed, two had incorrect facet aliases, and three had non-exact model-copied quotations. That output is **not** a complete or independently verified answer.
- The live model now **selects a pre-registered `quote_anchor_id`** rather than copying exact PDF text. The application resolves the selected ID to an exact source quote and runs the original strict citation, numeric and page validation. An invented or cross-source anchor is rejected.
- When the *identical* chunk appears in multiple retrieval facets, the code safely remaps its reference to the registered ID for the requested facet **only if chunk ID, page, citation URI, and source quote all match**. Mere topical similarity and matching pages are never enough.
- Prompt context is bounded dynamically to retain every planned facet and source quote selection within the existing 18k-character budget. Legacy `{evidence_id, quote}` output remains supported for custom providers.
- Fixed-structure tests cover full three-facet fake synthesis, forged anchors, cross-page and cross-facet rejection, budget pressure with five facets, numeric guard retention, and no extra remote calls. No live model run was made during this patch.
- **API:** preview, doctor, pytest = zero paid LLM calls; one opt-in live synthesis request with deterministic planner. All generated claims still require independent semantic and PDF source review; a successful quote match is NOT scientific validation.

## v1.4.2 — Phase 3.5.2: environment-independent doctor regression tests

- Fixed a test-only failure caused by assuming `live_research_available` must be false even when a user has configured `OPENAI_API_KEY` in `.env` and installed LangGraph.
- Explicitly test no key, environment-provided synthetic key and `.env`-provided synthetic key. Every test confirms doctor output never includes the key.
- Preserve the production behavior: `live_research_available` reports **local configuration readiness** (key present + LangGraph installed), not a verified API connection or validated research answer.
- No extra LLM calls, no change to the answer/citation validation runtime, and no real API credentials included.

## v1.4.1 — Phase 3.5.1: Live citation validation diagnostics

- Report only bounded, safe reason codes and optional 1-based claim index for rejected model claims; never expose the raw answer or provider secrets.
- Recover strictly validated claims if a mixed batch contains unsupported ones, but always label the result partial, human-review-required and semantically unverified.
- No automatic retries or extra token charges; simplified model instructions aim for one exact-quote claim per research facet.
- Full project preserved with offline regression tests for fabricated quotes, wrong facet IDs, numeric fabrication, mixed valid/invalid claims, and sanitized diagnostics.

## v1.4.0 — Phase 3.5: opt-in quote-anchored answer generation

- Introduced `agent-answer` command: `--mode preview` runs *without* an API key or model call; `--mode live` requires `OPENAI_API_KEY` and makes at most one answer request (default deterministic planner).
- Added real LangGraph synthesis node after evidence assessment and bounded retry; abstains before calling a model when structural facets are missing.
- Added strict untrusted model JSON output validation, source ID/facet crosschecks, literal quote verification, numeric claim guard, and URL generation by trusted application code only.
- Added partial-coverage and explicit abstention statuses; all generated claims are flagged for manual factual review. No semantic entailment checking is claimed.
- Added fake-model tests (no paid calls), optional native LangGraph tests, local safety tests for HTTP endpoints and redacted failures; documentation and doctor updated.
- Previous CLI commands, retrieval behavior and original paper artifacts remain unchanged.

## v1.3.0 — Phase 3.4: bounded structural evidence assessment

- Added deterministic per-facet evidence diagnostics (distinct primary chunks/pages, repeated content, source-fidelity warnings); explicitly no semantic/factual/contradiction validation.
- Added LangGraph `assess_evidence` and conditional bounded `retry_evidence` nodes without changing Phase 3.3 retrieval-only behavior.
- New `agent-assess` CLI, configurable minimum distinct hits and at-most-one alternate query per under-covered facet; tool-call budget, provenance and sanitized error paths remain enforced.
- New offline unit/integration and optional native LangGraph tests; updated quality gates, architecture and user instructions.
- No answer generation, no external API key needed in the default mode, no automatic change to retrieval weights or source relevance labels.

## v1.2.0 — Phase 3.3: real LangGraph tool integration

- New `agent-research` CLI and auditable LangGraph execution graph.
- Real bounded adapters for arXiv metadata, PDF ingestion, chunking, local embeddings, FAISS index verification, and hybrid retrieval.
- Offline/cache-first by default; external arXiv network access is explicit and does not use an API key.
- Typed execution configuration, tool budget, structured safe events, source hash checks and per-query citations.
- Existing `agent-preview` remains plan-only; answer synthesis and evidence sufficiency remain unimplemented.
- New offline and native test suites plus documentation and quality gates.

# Changelog

## v1.1.0 — Phase 3.2 Question-Aware Research Planning

- Upgraded the LangGraph planning node from a fixed scaffold to deterministic multi-intent analysis and bounded subquestion decomposition.
- Added Pydantic-validated research task DAGs with explicit dependencies, tool proposals, expected outputs, and `not_executed` state.
- Added optional, user-selected OpenAI-compatible LLM subquestion suggestions with strict schema/safety checks, sanitized failures and deterministic fallback; no agent research tools or answer generation.
- Added 12-case synthetic planning regression suite and `evaluate-planning` CLI, producing JSON/Markdown reports with explicit limitations.
- Preserved retrieval, evaluation, optimization, Phase 3.1 graph topology and file-existence preflight. No automatic RRF parameter deployment.


## v1.0.0 — Phase 3.1: LangGraph Foundation

- Introduced an **optional real LangGraph StateGraph** with typed state, graph edges, and deterministic conditional routing.
- Added request validation, future-task outline, safe read-only local artifact preflight, three branch statuses and ordered node execution trace.
- Added `agent-preview` CLI command; zero LLM calls, zero external tool calls, zero answers, no checkpoint persistence.
- Preserved the entire previous discovery, ingestion, embedding, retrieval, optimization and evaluation pipeline unchanged.
- Added native LangGraph smoke tests (only when dependency installed), pure-node unit tests, graph topology tests, and CLI failure-path tests.
- Added opt-in `[agent]` dependency and CI installation for integration verification.
- Retained provisional PDF/qrel limitations and no automatic retrieval-weight deployment.



## v0.9.0 — Phase 2.7: Retrieval Optimization and Label Audit

- Added validated configurable weighted reciprocal rank fusion (preserving unweighted `(1,1)` as default).
- Added optional capped primary results per PDF page, disabled by default and source-citation preserving.
- Added a fixed 14-setting grid, deterministic query-level train/holdout split (6/2 for the existing eight questions), train-only selection, holdout diagnostics, and candidate caching across trials.
- Added immutable page-label review suggestions with explicit warnings; no automatic ground-truth edits or production weight changes.
- Added JSON/Markdown optimization reports with suite/corpus SHA-256 and model revision; kept old artifacts/CLI behavior backward-compatible.
- Extended offline synthetic tests for RRF, edge cases, page caps, leakage safety, reproducible reporting and CLI.
- **Known limitations:** PDF completeness and tables remain unaudited; all eight page-qrels are provisional. Tiny single-paper holdout does not establish generalization.

## v0.8.0 — Phase 2.6: Retrieval Evaluation

- Added a versioned, **provisional** eight-question FAIR-RAG page-relevance suite.
- Added reproducible three-way BM25 / FAISS / hybrid retrieval scoring with Precision@K, Recall@K, MRR@K, and Hit@K for K=1,3,5,10 (custom K values supported).
- Added per-query diagnostics, macro averages and local JSON/Markdown reports tied to source chunk SHA-256 and pinned embedding model revision.
- Reuse one query encoder instance across evaluation questions to avoid unnecessary model loads.
- Added offline scoring/validation/reporting regression tests and a CLI command.
- Clarified that metrics are **page-proxy ranking diagnostics**; labels are not independently adjudicated, source PDF fidelity remains unverified, and LLM agent synthesis is not yet implemented.

## v0.7.1 — Phase 2.5 regression patch

- Isolated a missing-embedding CLI test from the user's locally cached source data.

## v0.7.0 — Phase 2.5: Hybrid Retrieval

- Introduced BM25, FAISS/BM25 rank fusion (RRF), same-page context expansion and a four-name coverage diagnostic.

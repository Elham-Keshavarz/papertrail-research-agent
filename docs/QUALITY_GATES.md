# Current release gate — Phase 6 (v1.6.1)

**Prepared for GitHub, not published.** User verified that v1.6.0 installed and its local Streamlit server started on Windows with model weights loading. That does not establish full browser UX acceptance or independent retrieval/answer correctness. Before publishing, review `docs/GITHUB_RELEASE.md`, run `python tools/check_release.py` and `python -m pytest -q`, and inspect staged Git files and Git history. The scanner does not verify all possible secrets.

## Historical v1.4.3 gate (retained for context)

A real v1.4.2 model run produced just 1/6 mechanically accepted claims and 1/3 covered facets; this is not enough to assert general research-answer completeness. v1.4.3 removes the model's need to copy exact PDF lines by requiring a registered quote ID and adds safe duplicate-chunk alias handling. Offline tests pass, but **native LangGraph/FAISS integrations and a real v1.4.3 live model output remain unverified in this environment**. Do not publish an answer-accuracy percentage, semantic faithfulness claim or end-to-end success claim until independently evaluated. Cite/quote matching does not substitute for page fidelity and entailment review.

# Quality gates

## Definition of done

1. Implemented capability is testable without the public network.
2. Malformed, missing or untrusted inputs fail in a controlled way.
3. Secret values and raw article contents are not printed by diagnostics.
4. Documentation covers every new CLI operation and limitation.
5. A live smoke test is performed from the user's machine before marking external service integration verified.

## Completed

- [x] Phase 0: installable package, settings, CLI, CI, unit tests
- [x] Phase 1: arXiv metadata, XML parser, rate limits, mock tests
- [x] Phase 1 live discovery verified by user (two queries and successful results)
- [x] Phase 2.1: single-paper HTTP download and page-aware text extraction
- [x] Phase 2.1: bounded file size, page count, ID validation and redirect restrictions
- [x] Phase 2.1: document SHA-256 and local cache
- [x] Phase 2.1: offline tests with synthetic PDFs, including blank/encrypted/corrupt pages
- [x] Phase 2.1: user verified live arXiv PDF extraction (30/30 pages, 178,156 characters)
- [x] Phase 2.2: deterministic page-aware, provenance-preserving character chunker
- [x] Phase 2.2: exact text offsets, chunk overlap bounds, offline pipeline and regression tests
- [x] Phase 2.2: user successfully generated 198 chunks from a 30-page paper
- [x] Phase 2.3: pinned local model adapter and fully offline mocked tests
- [x] Phase 2.3: token truncation preflight, normalized float32 matrix, SHA-256-validated cache

## Verified on user's Windows environment (previous milestones)

- [x] Phase 2.3: 365-vectors local MiniLM embedding; manifest normalized and complete
- [x] Phase 2.4: 195/195 tests; native FAISS `IndexFlatIP` and cited semantic search
- [x] Phase 2.4.1: 202/202 tests; local 421-vector embedding and 421-entry FAISS index with `--chunk-size 400 --overlap 60`
- [x] Dataset-name presence in extracted JSON: HotpotQA, 2WikiMultiHopQA, MuSiQue, TriviaQA
- [ ] Content completeness/reading order still needs real PDF manual audit


## Phase 4.3 validation status

- [x] Exact quote/source-ID/facet/page/number guards with deterministic per-claim reject/review behavior.
- [x] Synthetic regression with deliberately misleading/negated paraphrases, no claims of automatic NLI.
- [x] Local manual review queue and user-entered decision summaries (without authentication).
- [ ] Independent human evaluation of PDF provenance and semantic entailment on multiple papers.
- [ ] Answer faithfulness/entailment accuracy measurement; cannot be inferred from these tests.

## Not implemented

- OCR, verified table/formula extraction, comprehensive reading-order validation
- Supervised rankers, manually labeled general retrieval benchmark, cross-paper search
- Automated semantic claim entailment verification, autonomous multi-step replanning and persistent checkpointing (opt-in cited LLM synthesis and bounded LangGraph tools are available)


## Future evaluation protocol

Make a frozen benchmark with labelled retrieval relevance and manually verified passage-to-paper/page relationships. Report Recall@K, MRR, supported-claim rate, tool costs and latency. Do not claim results that have not been measured.


## Phase 2.4 — FAISS search

- Query and document embeddings share the same pinned model revision and dimensionality.
- Exact `IndexFlatIP` cosine ranking matches a NumPy dot-product reference on synthetic fixtures.
- `chunk_id`, PDF page and character offsets are carried into the user-visible result without alteration.
- Tests cover stale or damaged cached indexes, missing embeddings, invalid queries, nonfinite/zero query vectors and a native FAISS smoke test (when installed).
- Native Windows end-to-end retrieval and manually judged query relevance are still user-run acceptance gates, not claimed offline success.


### Layout-aware ingestion quality gate

Synthetic multi-column PDF regression tests check reading-order reconstruction without mixing columns. Real PDFs still require reviewer spot-checks of page text; unit tests alone do not establish evidence fidelity.

## Phase 2.5 — Hybrid Retrieval

- [x] Deterministic Unicode lexical BM25 on verified, page-attributed chunks
- [x] RRF of dense/sparse rank lists without comparing incompatible score scales
- [x] Cited same-page neighbor expansion and readable per-method score provenance
- [x] Single-question literal benchmark-name coverage diagnostic (not full QA accuracy)
- [x] Offline synthetic integration and input-validation tests
- [x] Native FAISS + Hybrid four-name-coverage diagnostic on the user's article (FAISS 3/4, Hybrid 4/4, K=10, context=1)
- [ ] Manual review of benchmark and SEA passages for reading-order and completeness

## Phase 2.6 — Retrieval Evaluation

- [x] Page-attributed provisional question/ground-truth suite stored in version control (8 questions)
- [x] BM25, FAISS and Hybrid evaluation on identical queries/corpus
- [x] Precision@K, page-recall@K, MRR@K and Hit@K with explicit page-proxy limitations
- [x] Deterministic JSON and Markdown report with SHA-256/model provenance
- [x] Synthetic offline tests (duplicate pages, malformed suites, metrics, report output, CLI errors)
- [ ] User Windows native FAISS run of the 8-question evaluation
- [ ] Independent review of provisional page relevance labels against original PDF
- [ ] Full extracted PDF completeness and reading-order audit


## Phase 2.7 holdout/tuning gates

- `pytest -q`: verifies weighted-rank fusion, finite-parameter validation, source-aligned candidates, page diversity and reproducible report structure with synthetic data.
- `optimize-retrieval`: checks the **actual saved embedding corpus** and source manifest hashes and reports all train/validation candidates; no auto-deployment.
- Page relevance labels are `provisional`; generated unlabeled-page flags must be manually checked against original PDF pages.
- Eight queries from one paper (six train/two validate) are insufficient to assert measured generalization. Future GitHub claims require independently reviewed multi-paper relevance qrels and extraction/table-fidelity audit.


## Phase 3.1 LangGraph foundation gate

- [x] Deterministic validated request and arXiv ID handling; malformed inputs return controlled CLI validation errors.
- [x] Typed graph state, pure node tests, conditional routing and event trace with explicit non-execution markings.
- [x] Optional `[agent]` dependency, CI installation and native graph smoke tests (to run wherever LangGraph is available).
- [x] Read-only existence checks: **no claims of source integrity or answers**.
- [x] User Windows run of `pytest` with LangGraph installed and real `agent-preview` graph invocation (Phase 3.1).
- [ ] Real PDF layout/table completeness audit; independent page annotation adjudication.
- [ ] Future phases: discovery/retrieval tool integration, checkpointing, evidence validation, citations and answer generation. Phase 3.2 now includes optional model-proposed subquestions only.


## Phase 3.2 question-aware research planning gate

- Deterministic analysis of multiple question types; dependency-DAG schema validation, bounded subquestions and explicit `not_executed` steps.
- 12 synthetic labeled planning examples; JSON and Markdown regression report. These fixtures cannot establish external NLP / research quality.
- Opt-in OpenAI-compatible subquestion proposal: safe HTTPS endpoint, strict response structure, bounded questions, sanitized errors, deterministic fallback, provider secret kept outside graph state and output.
- No accidental provider request in offline mode; a real LangGraph native smoke test verifies branches and analysis when the `agent` extra is installed.
- Agent preview continues to report `tool_calls: 0`, `research_executed: false`, `answer_generated: false`. Only explicit LLM mode may count a model request.
- **Unresolved from Phase 2**: manually audit PDF extraction completeness and page-level qrels; results are provisional. External planning quality requires diverse, independently labeled questions and blind evaluation.


## Phase 3.3 real tool-execution gates (v1.2.0)

- [x] New `agent-research` command uses real existing tool adapters with a bounded LangGraph loop and no API key in default mode.
- [x] Network to arXiv never occurs without explicit `--allow-network`; source selection is review-first by default.
- [x] Cached chunk, embedding and FAISS alignment checks carried over; available PDF/extracted source hashes are compared. A hash match **is not** reading-order validation.
- [x] Tool limit, failed attempts, failure sanitization and citation/page mismatch checks covered with offline tests.
- [x] Plan and events distinguish cached versus executed stages; answer and assessment are not executed.
- [x] Real graph native tests are included and run in CI with `[agent]` extra.
- [ ] Native Windows `agent-research` smoke test using existing 421-vector cache; must be verified by user before claiming full end-to-end success.
- [ ] Manual review of PDF layout, extraction completeness, provisional qrels, cited passage reliability.
- [ ] Remaining: independent semantic verification, claim faithfulness evaluation, durable checkpoints and cross-paper synthesis (Phase 3.5 supplies cited output only).


## Phase 3.4 evidence-assessment gates (v1.3.0)

- [x] Independent per-subquestion distinct primary chunk counts; one well-covered facet never hides a missing one.
- [x] Explicit nonsemantic nature of structural thresholds; factual sufficiency is `null`, and no contradiction or answer verification is claimed.
- [x] Cross-facet repeated chunks, per-page concentration and source-audit warnings are exported.
- [x] Bounded, user-question-only retry when a facet has too few distinct primary chunks; no repeated identical queries; retry attempts consume remaining tool budget.
- [x] Forged primary or adjacent context citations are rejected; tool exception messages are redacted.
- [x] Previous `agent-research` remains retrieval-only; new `agent-assess` never writes an answer.
- [x] Offline regression tests cover retry success/failure/no-progress, tool budget, invalid settings, plan markings, and no-auto-network behavior.
- [ ] Native LangGraph and FAISS end-to-end tests on user's Windows with installed `agent/search` extras.
- [ ] Manual PDF reading-order and table-fidelity audit, source-label review, and independent semantic evidence evaluation.

## Phase 3.5 — cited answer release gates (`v1.4.1`)

- [x] Preview is no-key and executes zero answer-model calls.
- [x] Live mode requires a configured key **before** starting retrieval and is explicit opt-in.
- [x] LangGraph synthesis follows evidence assessment and bounded retries, never bypassing structural gaps.
- [x] Machine-readable model claims validated against a bounded source register.
- [x] Source URLs generated by application from validated arXiv ID/page (not model-generated URLs).
- [x] Verbatim quote and same-facet source-ID checks; numeric claims may not introduce unseen numerical data.
- [x] Invalid provider output, forged citations, unavailable model and empty evidence safely return no answer.
- [x] Redacted provider errors; no secrets in CLI logs / project ZIP; unit tests use fake providers.
- [ ] **Native LangGraph + FAISS full pipeline on end-user Windows with real existing cached paper**; user runs `pytest` and `agent-answer --mode preview` to confirm.
- [x] A user-owned live-provider request was received, but its model claims failed strict citation validation (v1.4.0); non-secret diagnostics are now provided in v1.4.1.
- [ ] **Live citation-compliant response with a user-owned API key** (one request; may incur a charge; no automatic retry).
- [ ] Manual source-PDF reading-order and table verification; citations are not yet independently checked for factual entailment.
- [ ] Independent human-rated answer faithfulness, coverage, citation accuracy and contradiction benchmark; do not publish unreviewed QA correctness metrics.
- [ ] Multi-paper synthesis and durable checkpoints remain future milestones.


## Phase 4.1 — offline provisional Agent diagnostics (v1.5.0)

- [x] Inspect exact logged evidence IDs **before** source register selection (retrieval stage).
- [x] Inspect which of those references survive into the **model-visible** register.
- [x] Evaluate k=3 versus k=5 using identical no-key deterministic preview settings.
- [x] Replay only a sanitized summary from an **earlier** Live run, clearly separating historical from new LLM calls.
- [x] Explicitly label all anchors as **provisional log-derived**, not complete expert relevance judgments.
- [x] Produce JSON + Markdown reports under ignored `data/` with no raw model provider secrets.
- [ ] Independent PDF extraction/page audit and human relevance/entailment annotations (future Phase 4.2).
- [ ] Held-out multi-paper evaluation with independently verified Precision/Recall/MRR/Faithfulness (future Phase 4.2+).
- [ ] Robust answer quality / contradiction / refusal / provider cost benchmarks (future Phase 4.3+).

The reported `observed_anchor_*_fraction` is **not Recall@k or Accuracy**. The underlying qrels are neither exhaustive nor independently validated.

### Phase 4.1.1 — Native failure visibility (v1.5.1)

If `agent-evaluate` fails on a real cached research run, the CLI emits a sanitized diagnostic with the case, k, `status`, failed tool and exception **type**, and the last graph node. Tests verify that arbitrary provider text and exception details are never echoed. The evaluator remains fail-closed: a failed run is not silently scored as a completed pipeline. Native third-party packages should be tested on the target Windows environment; optional tests may skip where those packages are absent.

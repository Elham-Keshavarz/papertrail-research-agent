# Phase 4.3 — Evidence-anchored claims and semantic review (v1.5.5)

## What this milestone **does**

1. Recheck accepted, quote-anchored claims against the in-memory evidence register after the existing strict validator. Invalid source IDs, step association, page, URL, quote presence and introduced numbers fail closed. This is a **mechanical source-chain check**.
2. Classify each accepted claim as `verbatim_within_quote` (its entire text literally occurs in a cited quote, after whitespace normalization) or `quote_anchored_nonverbatim` (a source quote exists, but the claim is not a literal extract). Both remain `semantic_support: not_evaluated`; **neither is a verified entailment judgment**.
3. Add a `synthesis.faithfulness_audit` object to a real, explicitly opt-in `agent-answer --mode live` result **without adding a model call**. It is attached only after a draft passes the existing citation validator. Live mode still needs a configured API key and may cost money; this phase does not request you run it.
4. Run **ten synthetic** regression cases offline, including exact extraction, plausible paraphrase, a negation inversion deliberately left for human review, unsupported numerical content, fabricated quotes, wrong-facet citation, selective rejection and abstention.
5. Audit a locally supplied draft + evidence register and create a **local, human-editable review queue**, with citations and the exact quoted text. Summarize the reviewer's decisions without treating them as independently validated ground truth.

## What this milestone **does not**

- It does **not** implement a semantic NLI/entailment verifier, contradiction detector, LLM-as-judge or independent PDF audit.
- An exact quote found in a **supplied register** is not proof that it was extracted correctly from the actual arXiv PDF. Check source pages manually when reviewing real answers.
- Synthetic regression pass rates are **not** faithfulness accuracy, scientific answer quality or published benchmark results.
- `baseline` remains the production/default Query strategy; `focused`, `multi` and `budgeted` remain opt-in offline-only. The v1.5.4 Source Register cap and its FAIR-RAG limitation are not silently changed.

## Windows — install from the full ZIP

Extract `papertrail_phase4_3_faithfulness_review_full_v1_5_5.zip`, open the **inner** `papertrail-research-agent` directory. To retain your locally cached article, chunks, embeddings and FAISS index, copy only data from v1.5.4 (assuming both version folders live next to each other in Downloads):

```powershell
Copy-Item -Path "..\..\papertrail_phase4_2_2_budget_aware_fusion_full_v1_5_4\papertrail-research-agent\data" -Destination ".\data" -Recurse -Force
Test-Path .\pyproject.toml
Test-Path .\data
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,search,agent]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m papertrail doctor
```

### Step 1 — free, offline regression (no downloaded model weights needed)

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-faithfulness-evaluate
Get-Content -Encoding UTF8 .\data\agent_evaluations\faithfulness_synthetic_v1_phase4_3_faithfulness.md
```

Expect `synthetic_regression_complete`, `case_count: 10`, `checks_passed: 10`, `checks_failed: 0`, `llm_calls: 0`. Report refers to **mocked evidence**, not FAIR-RAG answer verification. All real answer claims remain unreviewed.

### Step 2 — demonstrate the manual semantic-review workflow entirely offline

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-faithfulness-audit --input benchmarks/faithfulness_manual_bundle_example.json --name my_first_review
Get-Content -Encoding UTF8 .\data\faithfulness_reviews\my_first_review_review_queue.json
```

The file contains two **synthetic** accepted claims: one exact quotation and one paraphrase, both with `reviewer_decision: "pending"`. Edit that local JSON file to change `reviewer_decision` to **`supported`**, **`unsupported`**, or **`insufficient_context`**; leave `pending` if you cannot tell. Add a `reviewer_note` pointing to the independently checked PDF page and reason for the judgment.

Then summarize your locally entered decisions:

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-faithfulness-review-summary --review-file .\data\faithfulness_reviews\my_first_review_review_queue.json
```

The summary reports *user-entered decisions*, **not** verified independent labels. It does not print claim text or quotes to the console. Review queue files are local, potentially sensitive artifacts under `data/` (gitignored). An existing queue is **never overwritten**; choose another `--name` for subsequent runs.

### Step 3 — real offline audit (when a real draft and register exist)

Produce a **local JSON bundle** with exactly three fields:

- `plan`: the validated plan containing `id`, `proposed_tool: hybrid_retrieval`, and optionally `search_query`.
- `register`: the evidence register obtained by `build_evidence_register`, including source-ID, step, page, citation, and literal `passages`. Do **not** use the CLI's summarized `source_register`: it omits raw passages and cannot independently verify quotes.
- `draft`: the original untrusted structured model output with a `claims` list and exact `quote` text; or reuse the synthetic example and replace it with your own **local** evidence. When using model-selected `quote_anchor_id`, supply the matching quote options in the register; the existing resolver handles those.

Use `agent-faithfulness-audit --input path/to/local_bundle.json --name paper_02`. The validator rejects forged quote text and numerical additions; it does **not** determine whether a paraphrase logically follows from the quote. The provided real PDF must still be checked separately. Do not copy real private source text or keys into GitHub.

## Output field reference

- `citation_validation_status`: existing quote validator's state (`cited_answer_unverified`, `partial_answer_unverified`, or `citation_validation_failed`).
- `verbatim_within_quote_count`: exact, literal claim-substring presence in the supporting quote; not NLI.
- `quote_anchored_nonverbatim_count`: quote anchors valid but semantic support not evaluated.
- `rejections`: safe constant-coded rejection reasons from the existing validator; these claims are **not** included in the review queue.
- `human_review_required: true`, `semantic_entailment_verified: false`, `independent_pdf_provenance_verified: false`: hard safety contract.

**Exit criterion:** all synthetic checks pass, and review of at least one small real multi-paper sample is planned before any empirical accuracy claim. Next: Streamlit interface, then GitHub release. Avoid cycling back to repeatedly hand-tune FAIR-RAG retrieval.

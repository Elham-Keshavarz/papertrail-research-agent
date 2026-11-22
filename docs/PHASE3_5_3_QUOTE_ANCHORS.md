# Phase 3.5.3 — Safely Select Registered Source Quotes (`v1.4.3`)

## Why this patch exists

The first real `agent-answer --mode live` run under v1.4.2 returned `partial_answer_unverified` despite three structurally covered facets. Only **one of six generated claims** passed validation. The five rejections were: two `citation_id_or_facet_invalid` and three `quote_not_exact_in_source`. The model repeatedly paraphrased source quotes or used an evidence ID belonging to a different facet, even when the underlying chunk was repeated in the other facet. The partial answer was correctly labeled, but this is not adequate comprehensive research output.

## Fix without weakening citation protection

1. The system pre-computes at most **two** bounded, verbatim `quote_options` for each registered evidence item. IDs are stable for that run, e.g. `E03Q1`.
2. The model receives those choices and the surrounding read-only source excerpts, then emits a claim with `evidence_id` and `quote_anchor_id`. It does **not** need to copy a line-wrapped or hyphenated quote.
3. The application resolves each ID to its own already-registered exact quote, checks the quote's source, page, numerical content and facet, and constructs the citation URL itself. Wrong/unknown anchors are rejected with `quote_anchor_id_invalid`.
4. A mistaken **facet alias** is remapped only when the same chunk ID, same paper/page URL and exact supporting quote are registered for the requested facet. No matching by semantic similarity, page proximity, or a loose paraphrase.
5. The existing single-request model cap, strict JSON shape, claim-count bound, safe diagnostics, partial-recovery behavior and human-review flags remain. Legacy exact-string quotes from custom providers remain accepted as before.
6. Long PDFs use a bounded source-context view to fit under the original prompt character limit without silently dropping research facets or registered quote options.

## Boundaries and remaining work

- Anchored quotes are **verbatim substrings**, not independent semantic entailment verification. A plausible claim can still overstate or misinterpret a valid quote. `semantic_entailment_verified: false` and `human_review_required: true` remain mandatory.
- Source PDF layout, full table accuracy and source-page fidelity have not been independently audited.
- A live model may still choose an unsuitable quote, omit a facet, or produce malformed JSON. There is **no hidden model retry**.
- Phase 4 must benchmark real question coverage, citation correctness, abstention behavior and factual support with independent human review. The prior one-of-six-claims result is a baseline, not a performance claim.

## Running on Windows — one PowerShell command at a time

Extract the **full ZIP** and open its inner `papertrail-research-agent` directory, then run:

```powershell
Copy-Item -Path "..\..\papertrail_phase3_5_2_doctor_test_isolation_full\papertrail-research-agent\data" -Destination ".\data" -Recurse -Force
```

```powershell
Copy-Item -Path "..\..\papertrail_phase3_5_2_doctor_test_isolation_full\papertrail-research-agent\.env" -Destination ".\.env" -Force
```

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
.\.venv\Scripts\python.exe -m papertrail agent-answer --paper-id 2510.22344 --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --top-k 3 --mode preview
```

**API key:** None required for the above commands; running tests never contacts an LLM provider. A locally configured `.env` may be reused, but do not paste it into a chat, commit it or include it in the ZIP.

Once those pass, optionally perform **one** Live experiment (the `--mode live` flag below **uses your configured API key, sends evidence to the provider, and may incur a charge**):

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-answer --paper-id 2510.22344 --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --top-k 3 --mode live
```

Inspect `synthesis.claim_count`, `synthesis.covered_step_ids`, `missing_step_ids`, `rejected_claims_count`, `rejection_reasons`, `llm_calls`, and `semantic_entailment_verified`. Even `cited_answer_unverified` requires checking the real PDF.

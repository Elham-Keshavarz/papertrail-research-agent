# Phase 3.5.1 — Safe Live Citation Diagnostics (v1.4.1)

The first real Phase 3.5 live execution confirmed that model requests worked (`llm_calls: 1`), but the model's proposed claim(s) were rejected with `citation_validation_failed`. Phase 3.5 hid the exact validation reason, making troubleshooting needlessly costly. This update keeps **strict quote verification** and makes rejection diagnostics actionable without revealing the model output, provider response, API key or sensitive content.

## What changes

- Exact source IDs and same-facet checks are still mandatory. Quotes must still be exact continuous spans of registered source text (whitespace/case normalization only). Numeric facts must appear in the corresponding quote(s).
- For an invalid model draft, output `synthesis.validation_rejections` includes fixed, non-sensitive `reason_code` values, and **1-based** `claim_index` where applicable. Examples: `quote_not_exact_in_source`, `citation_id_or_facet_invalid`, `numeric_claim_unanchored`, `claim_schema_invalid`, `claim_markup_rejected`.
- If at least one claim passes ALL existing quote, numeric, facet, citation and formatting guards, but others fail, the system publishes only accepted claims with `partial_answer_unverified`, `rejected_claims_count` and `rejection_reasons`. Never calls the model a second time automatically.
- The LLM prompt now prefers one concise citation-anchored claim per question facet, minimizing malformed or noisy drafts.
- No independent semantic entailment verification. **Even accepted claims remain unverified**, especially for figures and extracted PDF tables.

## Test commands (Windows PowerShell, one command per terminal run)

Run these from the new extracted `papertrail-research-agent` project root:

```powershell
Copy-Item -Path "..\..\papertrail_phase3_5_cited_answer_generation_full\papertrail-research-agent\data" -Destination ".\data" -Recurse -Force
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
.\.venv\Scripts\python.exe -m papertrail agent-answer --paper-id 2510.22344 --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --top-k 3 --mode preview
```

To run live, first **securely copy your existing `.env`** from the old project folder into the new root (do not include it in the ZIP or Git). Verify the provider URL and model name locally, then:

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-answer --paper-id 2510.22344 --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --top-k 3 --mode live
```

**API key:** offline tests and preview: **no key**. Live: **`OPENAI_API_KEY` required**. Each live run may incur cost; this change makes no additional request. Do **not** send `.env`, key or raw provider responses to anyone. For diagnostics, only `status` and `synthesis.validation_rejections` need to be shared.

## Diagnostic interpretation

| Reason code | Meaning |
|---|---|
| `quote_not_exact_in_source` | Model quoted a paraphrase or content absent from the registered excerpt |
| `citation_id_or_facet_invalid` | Source ID not in register or linked to a different subquestion |
| `numeric_claim_unanchored` | A numerical value was claimed but absent from the supporting quote |
| `claim_schema_invalid` | A claim object's keys differ from the required contract |
| `citation_schema_invalid` | Citation object's keys differ from the required contract |
| `claim_markup_rejected` | Model injected markdown, a link or another untrusted marker into the claim |
| `duplicate_claim` | Model proposed repeated, identical claims |
| `draft_schema_invalid` | Top-level JSON shape doesn't match the required format |
| `validation_rejected` | The problem was not covered by known diagnostic categories |

If every claim is rejected: `citation_validation_failed`, `answer: null` (no unverified answer shown). If some claims pass: `partial_answer_unverified`, never a complete answer even when all facets have a valid claim.

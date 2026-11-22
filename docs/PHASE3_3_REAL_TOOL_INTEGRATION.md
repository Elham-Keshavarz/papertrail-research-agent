# Phase 3.3 — Real Tool Integration (PaperTrail v1.2.0)

## Scope and boundary

This milestone adds **real, bounded tool invocations inside a genuine LangGraph `StateGraph`**. The workflow uses the existing Phase 1–2 paper discovery, arXiv PDF extraction, page-aware chunking, local MiniLM embeddings, hashed FAISS indexing, and BM25+FAISS RRF retrieval. It is not a research-answer agent yet: **no evidence-sufficiency assessment, cross-source fact verification, answer generation, or citation-supported synthesis** is implemented. Phase 3.4/3.5 add those capabilities.

Default mode does not call an LLM, does not send requests to arXiv, and does not need any API key. It does require the installed local SentenceTransformer model and FAISS backend if retrieval is to run. `agent-research` explicitly sets Hugging Face offline flags for the duration of an offline invocation, so *missing* model weights cause a controlled error instead of an automatic model download. Initial dependency/model installation may require internet; use earlier `embed-document` commands to pre-cache the pinned model if you want offline tool runs. The `--allow-network` opt-in permits arXiv requests and model weight resolution; no paid API key is involved unless `--planner llm` is separately selected.

## User commands (PowerShell, project root)

Transfer your existing *data* folder from Phase 3.2 to this project before running retrieval:

```powershell
Copy-Item -Path "..\..\papertrail_phase3_2_research_planning_full\papertrail-research-agent\data" -Destination ".\data" -Recurse -Force
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
.\.venv\Scripts\python.exe -m papertrail agent-research --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --paper-id 2510.22344 --top-k 3 --max-tool-calls 12
```

If you have the copied 421-vector cache, you should see a source-aware plan, `index_document` (possibly cached), up to 3 hybrid searches for the comparison/metrics/limitations facets, source-page citations, per-tool events, `llm_calls: 0`, `answer_generated: false`, and status `evidence_retrieved_no_answer`. **Report the actual output; this has not been independently executed on your Windows environment yet.**

### Network opt-in (public arXiv; NO API key required)

If no paper ID is supplied, default offline mode responds `needs_paper_selection` without contacting arXiv. To discover candidate papers, deliberately add `--allow-network`:

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-research --question "What are recent methods for faithful retrieval augmented generation?" --allow-network
```

Result `paper_selection_required` lists candidate IDs. The user should review title/ID and rerun using `--paper-id ...`. To explicitly select the *first* search result and continue (not recommended without checking relevance), supply `--allow-network --auto-select-first`. This may download a paper PDF up to 20 MiB; the model weights might also need downloading if not cached. Automatic first-result selection does **not** imply the paper is the correct source.

For a known paper without local artifacts, `--allow-network --paper-id 2510.22344` may download and process its PDF. The pipeline picks character chunks of 400 with overlap 60 for newly processed papers; the tokenizer preflight **rejects** any passages longer than the pinned model's allowed input tokens rather than silently truncating them. The step fails explicitly if new paper processing requires different chunk settings; the existing `chunk-document` CLI can be used with smaller values before resuming.

### Optional LLM planning (API key required)

`--planner llm` opts in to external OpenAI-compatible model calls for proposing subquestions. Configure `OPENAI_API_KEY` in a local, gitignored `.env`, plus `PAPERTRAIL_MODEL_NAME` and optionally `PAPERTRAIL_OPENAI_BASE_URL`. **This is optional; not used by default or required by this phase.** Without a key it falls back to deterministic planning. **Never paste or commit your API key.** Depending on the provider this may incur charges. The `--allow-network` switch controls **arXiv discovery/PDF access**, not this separately authorized LLM proposal.

## Real graph and execution semantics

```text
START
  -> validate_request
  -> plan_research (deterministic default, optional LLM proposal)
  -> select_source
      -- ID provided -------------------> prepare
      -- no ID, offline ---------------> finalize (needs_paper_selection)
      -- no ID, --allow-network -------> discover
             -- candidates, no auto --> finalize (paper_selection_required)
             -- explicit auto select --> prepare
  -> prepare
      if needed: PDF ingestion / chunking / embedding
      always: index tool (verifies chunk/vector hashes; verifies/rebuilds index cache)
      verify PDF/extraction hashes if both source files exist
  -> retrieve (one validated subquestion per loop; bounded by tool budget)
  -> finalize (no synthesis)
END
```

The `plan` reflects which stages executed (`completed`) or reused existing artifacts (`cached`) and leaves `assess` and `answer` as `not_executed`. A network call is **never** implicitly enabled by selecting a paper ID; explicit `--allow-network` is required if a new PDF must be fetched. Existing PDF files may be re-extracted offline. Existing local embeddings are validated for consistency with chunk IDs/source hashes before FAISS usage; corrupted or stale caches abort or rebuild from valid local dependencies rather than silently cite mismatched passages.

- `tool_calls` counts **attempted actual tool invocations**, including failures and validating a cached FAISS index. Reused artifact skips are recorded but **not** counted as tool calls. The default max is 12; valid range is 1–20.
- `llm_calls` is separate, zero by default.
- `tool_events` records tool name, status, and small safe metadata; raw provider exceptions are **never** printed (only exception type).
- `evidence` groups primary retrieved passages per subquestion with page number, chunk ID, RRF rank/score, exact arXiv citation, and adjacent **same-page** context. These are raw source excerpts, not established facts.
- `source_audit.pdf_extraction_quality_verified` remains **false** even when SHA-256 chains match: a hash match doesn't establish correct column reading order or faithful tables.
- `errors` and `status` indicate bounded failures; the command returns exit code 1 on tool failures, 2 on invalid flags/configuration.

### Bounded error cases

- No `--paper-id` in offline mode: `needs_paper_selection`, zero tool calls.
- Missing local PDF/extracted text and no `--allow-network`: `offline_source_missing`, zero network requests.
- arXiv search with no matches: `no_papers_found`.
- arXiv search results without explicit auto-selection: `paper_selection_required`.
- More requested operations than `--max-tool-calls`: `tool_budget_exhausted` (no extra tool invocation).
- PDF / embeddings / FAISS / search failure: `tool_failed`, with sanitized error category.
- Mismatched cited arXiv paper/page in adapter output: fail closed before adding that evidence to the report.
- All queries return no passages: `no_retrieval_hits` rather than claiming that evidence was found.

## Evaluation and quality limitations

Offline unit tests with fake tool adapters exercise graph routing, bounds, cache reuse, source provenance, safe error handling and citation checks. Optional **native LangGraph tests** run on the user system/CI where `[agent]` is installed. The core PDF, embeddings and FAISS components are also covered by independent integration tests from previous milestones.

End-to-end running on the user's cached 421-vector paper remains an explicit Windows acceptance gate. These retrieval results are **not** measured answer accuracy, independent paper-level relevance assessment, or source-grounded LLM synthesis. The 8-item relevance suite has provisional page labels; manually audit reading order and table fidelity before claiming scientific retrieval precision.

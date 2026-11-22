# Phase 3.4 — Evidence Assessment (v1.3.0)

## Scope

`agent-assess` invokes the existing Phase 3.3 bounded LangGraph paper retrieval pipeline and then enters a **new explicit `assess_evidence` node**. Assessment produces independent per-subquestion diagnostics and, if necessary, one optional retrieval retry per under-covered facet. **No LLM, API key, answer writing or semantic evaluator is required or used by default.**

The structural proxy counts distinct primary `chunk_id`s per planned retrieval step. By default the threshold is `min(2, --top-k)`. It also records primary page diversity, across-facet chunk reuse, warnings for multiple hits on one page, source-chain audit state and whether the extracted PDF's reading order has undergone independent verification (currently `false`). Evidence values are not claims of truth, relevance, completeness, entailment or contradiction-free content. The outcome string `evidence_assessed_unverified` means **only** that the structural threshold was reached.

## Architecture and control-flow

```text
validate_request -> plan_research -> select_source -> [discover?]
  -> prepare (cache / PDF -> chunks -> embeddings -> FAISS)
  -> retrieve (one bounded hybrid query per subquestion)
  -> assess_evidence
      |- structurally covered -> finalize
      |- gaps + allowed retry -> retry_evidence -> assess_evidence
      |- gaps + exhausted/disabled retry -> finalize
  -> final result (NO answer)
```

The planner's original DAG is preserved; `assess` is labeled `completed_structural_unverified` after assessment. `answer` is still `not_executed`. Assessment is deterministic and does **not** count as an external tool call. Each actual retry is an additional bounded hybrid retrieval tool call, counted even when it fails. The tool budget is enforced before every call. All attempted retrievals are in `tool_events` with safe metadata; raw provider exceptions and API keys are never included in result objects. The default retry strategy removes planner-added wording by issuing only the **user's original question** (no invented facts or PDF instructions). The strategy is intentionally limited to **one alternate query per missing facet** to prevent useless repeated calls.

A repeated query is skipped, not sent again; at most one retry per facet is currently supported. No automatic discovery/download is allowed without the separate `--allow-network` opt-in. In offline mode the Hugging Face backend must find model weights cached locally.

## Windows commands (project root, one per block)

Copy the complete `data` folder from the Phase 3.3 directory:

```powershell
Copy-Item -Path "..\..\papertrail_phase3_3_real_tool_integration_full\papertrail-research-agent\data" -Destination ".\data" -Recurse -Force
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
.\.venv\Scripts\python.exe -m papertrail agent-assess --paper-id 2510.22344 --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --top-k 3
```

Optional experiment with no retry:

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-assess --paper-id 2510.22344 --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --top-k 3 --max-refinement-rounds 0
```

No need to run the LLM planner. Using `--planner llm` explicitly requires a provider API key (`OPENAI_API_KEY` in local `.env`), and may cost money. The `--allow-network` flag permits public arXiv requests/model downloads, **not** LLM planning on its own.

## Result fields and interpretation

- `assessment.decision`: `structurally_covered_unverified` / `needs_more_retrieval` / `no_retrieval_steps`. This is **not** scientific certainty.
- `assessment.per_subquestion[*]`: exact primary-hit count, distinct chunk count, distinct source pages, threshold pass/fail, retry count, and `semantic_relevance_verified: false`.
- `assessment.gap_step_ids`: facets with fewer distinct primary chunks than required; their absence cannot be offset by another facet's many chunks.
- `assessment.cross_subquestion_duplicate_chunk_ids`: repeated chunk IDs across facets; no claim that repeated passages are incorrect.
- `assessment.warnings`: page concentration, duplicated chunks, and missing independent PDF extraction audit.
- `assessment.factually_sufficient: null`, `semantic_coverage_verified: false`, `contradiction_analysis_performed: false`, `answer_generation_allowed: false` are intentional restrictions.
- `retry_attempts`, `retry_justification`, `tool_events`: exactly which facets were retried, why, what safe query was used, and what happened.
- `source_audit`: hash and source-chain status from the Phase 3.3 tool execution. Hash agreement does not prove extraction completeness.

A structurally covered result can be irrelevant or misleading. Especially for the one-paper FAIR-RAG demo, all relevance labels and original two-column PDF extraction need manual audit. The original Phase 2.6/2.7 tiny one-paper benchmarks did not validate general retrieval accuracy. Phase 3.5 will require a separately authorized LLM provider/API key for full answer synthesis; it must validate citations and refuse unsupported claims, and should be tested against independently reviewed question/evidence labels.

## Quality gates

- Preserve Phase 3.3 behavior under `agent-research` and previous model/index/data formats.
- Run every pure assessment and graph routing test offline, including no hits, duplicate chunks, missing facets, forged citation, failed tool, no novel retry results, tool-call budget, network opt-in and no answer generation.
- Run real LangGraph native smoke tests when the `[agent]` extra is installed, plus FAISS native tests with `[search]` extra. Local sandbox installations may skip these if modules are unavailable. Windows/CI must confirm native tests.
- Never treat structural thresholds as semantic or factual evidence. Preserve source page/hash/fidelity warnings in output.
- No automated source PDF or provisional relevance label modifications.

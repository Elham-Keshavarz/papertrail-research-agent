# Phase 3.5 — Quote-Anchored Cited Answer Generation (`v1.4.3`)

This phase extends real LangGraph tool execution and cautious per-facet evidence assessment with an **explicitly opt-in** answer node. The model can propose claims, but **the application**, not the model, constructs citations from registered real retrieval hits. Quote anchoring is checked mechanically. **Semantic entailment, PDF reading-order quality and claim factuality are NOT independently verified.** The output requires human review before academic use.

## Model-selected quote anchors (v1.4.3)

When using the built-in live generator, each piece of retrieved evidence includes bounded `quote_options` containing authentic source substrings with IDs such as `E01Q1`. Instead of transcribing a complicated PDF snippet exactly, the LLM must choose `quote_anchor_id` and `evidence_id`. The application resolves the ID to the real quote, checks that the original source contains it, and produces an arXiv page link itself. Quotes/claims are never trusted solely because a model emitted them.

Custom provider integrations using the earlier `quote` string field remain valid. Mis-typed and forged anchor IDs fail closed. Strict alias repair is permitted only for identical registered chunks on the same source page and URL. Neither a valid quote nor an apparently sensible claim proves entailment, lack of contradictions, or PDF extraction fidelity.

**Observed user Live baseline (v1.4.2):** 1/6 drafted claims retained, 1/3 question facets covered. This v1.4.3 patch does not yet establish better real-model outcomes; re-evaluate with independent benchmark questions in Phase 4.

## API key rules — read BEFORE running live mode

| Command | OpenAI-compatible API key | May incur API charges? | Network notes |
| --- | --- | --- | --- |
| `python -m pytest -q` | No | No | No live model calls; simulated providers |
| `python -m papertrail agent-answer --mode preview ...` | No | No | By default offline; uses cached local evidence |
| `python -m papertrail agent-answer --mode live ...` | **Required: `OPENAI_API_KEY`** | **Yes** | Sends question + bounded cited excerpts to configured third-party model endpoint |
| `--planner llm` (optional, LIVE only in agent-answer) | Required | Yes, additional request | Planner fallback exists; `--planner deterministic` avoids it |
| `--allow-network` | Not for arXiv | No paid arXiv API | Opts into paper discovery/PDF downloads and possible model-weight downloads; independent of `--mode live` |

For normal first-time testing, use `--mode preview`; **do not create or populate a key yet**.

When ready for a real answer:

1. In the project's root directory, copy `.env.example` to `.env` (`Copy-Item .env.example .env`).
2. Edit `.env` with a text editor (`notepad .env`) and fill in exactly these values, using your provider's own details:

   ```dotenv
   PAPERTRAIL_MODEL_NAME=gpt-4.1-mini
   PAPERTRAIL_OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_API_KEY=PASTE_YOUR_KEY_LOCALLY_HERE
   ```

3. Keep `.env` private. `.gitignore` excludes it. **Do not upload the file to GitHub or paste the key into chats or screenshots.** The default model is an example; your key must have access to the selected model, and model names / billing depend on the provider. For a compatible third-party service, change *both* base URL and model name to the provider's documented values. A remote provider requires HTTPS.
4. Run one live command; the default deterministic planner makes no separate LLM request. **There is at most one answer-generation HTTP request**; errors are not automatically retried. Consult provider pricing before running live mode.

## Windows quick start — one command at a time

Extract the full Phase 3.5 ZIP to a new folder, open the **inner** `papertrail-research-agent` in VS Code. The previous Phase 3.4 folder and its `data` directory must exist at the expected location.

```powershell
Copy-Item -Path "..\..\papertrail_phase3_4_evidence_assessment_full\papertrail-research-agent\data" -Destination ".\data" -Recurse -Force
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

Expected first-run `status`: `ready_for_llm_synthesis` if there are enough distinct chunks to meet the **structural** threshold, or `insufficient_evidence` if not. Expect `llm_calls: 0` and `answer_generated: false` in preview. Source excerpts will not be automatically sent to a model.

When and **only when** you have a configured provider and have verified the preview output, run:

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-answer --paper-id 2510.22344 --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --top-k 3 --mode live
```

Possible outcomes: `cited_answer_unverified`, `partial_answer_unverified`, `insufficient_evidence`, `citation_validation_failed`, `llm_unavailable`, or an upstream tool failure. **Successful-looking text is not certified factual.** Provider errors and bad claim formats are reported without raw model responses / provider credentials. A `citation_validation_failed` result does not return the rejected model's claims. In v1.4.1, `synthesis.validation_rejections` reports only fixed reason codes and claim indices; an independently valid subset of a mixed draft can be returned as `partial_answer_unverified` with a count of rejected claims. No additional model call is made.

## Workflow and trust boundary

```text
User question
  -> validated request / planned retrieval facets
  -> source selection / cached PDF, chunks, embeddings, FAISS index
  -> 1..5 bounded hybrid retrieval tasks
  -> deterministic evidence coverage + limited retry
  -> is structural evidence sufficient?
       No  -> insufficient_evidence; no model call
       Yes -> build registered source ID E01..E12
                -> mode preview: source register; NO answer/model call
                -> mode live: ONE configured model call
                     -> strict JSON claims + source IDs + literal quotes
                     -> validation of source ID, facet, quote, numbers and markup
                     -> render application-generated Markdown citation links
                     -> mark ALL outputs semantically unverified
```

The LLM cannot directly choose the citation URL, alter the source register, add unregistered source IDs, cite a different facet, or include markup/links in a claim. Numerical claims must be present in quoted text. **This is a useful guard, not a semantic factuality checker.** A misleading claim can still contain a real quote from a valid page, and the PDF layout heuristic itself still needs manual inspection. In particular, evaluation figures and tables should not be trusted without source-page review.

### Limiting model exposure

- At most 12 registered hits, with bounded primary excerpts and same-page neighbor context.
- One request per run in default `--planner deterministic --mode live` (two at most if opting into `--planner llm` plus the answer call).
- Explicit key and live switch; no background calls, retries or autonomous loops.
- Question and excerpt text are sent to the configured third-party provider **only in live mode**, so check data-handling terms if using sensitive sources.
- No raw provider error bodies in user-visible traces. Only trusted arXiv-derived links are emitted.

### Distinct statuses

| Status | What it means |
| --- | --- |
| `ready_for_llm_synthesis` | Preview generated a source register, no LLM call, **no answer** |
| `cited_answer_unverified` | At least one anchored cited claim for each planned facet; still not fact-checked |
| `partial_answer_unverified` | Some facets lack cited claims; partial answer with explicit missing-facet list |
| `insufficient_evidence` | Structure insufficient, or model returned no claims; explicit abstention |
| `citation_validation_failed` | Claims, source IDs or quotes failed validation; rejected answer is discarded |
| `llm_unavailable` | Provider failure or invalid model output; no answer; one attempt at most |

This milestone does **not** implement cross-paper synthesis, independent semantic/contradiction verification, persistent checkpoints, source-table parsing or answer-quality benchmarking on independent human-rated cases. These remain future work and release quality gates.

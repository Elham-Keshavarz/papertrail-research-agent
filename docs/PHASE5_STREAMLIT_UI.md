# Phase 5 — Streamlit User Interface (PaperTrail v1.6.0)

## Scope

This is a **real local UI over the existing LangGraph execution pipeline**, not a mock chat interface. It supports:

1. **Research:** select locally indexed arXiv paper, enter a question, run offline retrieval with Baseline/Focused/Multi/Budgeted, inspect evidence pages, facet statuses and tool trace. Preview **does not generate any LLM answer**.
2. **Live synthesis:** separately configure an OpenAI-compatible model in your local `.env`, choose **Live + Baseline** and explicitly approve a potentially paid request. Only pressing **Run PaperTrail** initiates the model call. Quote validation and numeric-reference checks are enforced in the old research engine; do **not** mistake them for semantic entailment or factual accuracy. Live results cannot use experimental query policies.
3. **Paper library:** inspect local artifacts, opt into a bounded public arXiv discovery query, select a reviewed result, and explicitly download/chunk/embed/index it. Model weights may also download on first import; do not enable this workflow in an offline or confidential environment. The original PDF and research data stay locally in `data/`.
4. **Human review:** save the existing Phase 4.3 mechanical faithfulness audit into a private queue with claim text and verified-in-register quotes, then manually record `supported`, `unsupported` or `insufficient_context`. Local reviewer decisions have **no independent verification**.

The UI server is intended for **single-user localhost only**, not Internet exposure. No accounts, multi-user authentication or remote deployment are supplied.

## Windows installation

Extract `papertrail_phase5_streamlit_full_v1_6_0.zip`. In VS Code open the **inner** `papertrail-research-agent` directory. Run the following commands in PowerShell from that directory.

Copy your existing local cache from v1.5.5 into the new version (never copy `.venv`):

```powershell
Copy-Item -Path "..\..\papertrail_phase4_3_faithfulness_review_full_v1_5_5\papertrail-research-agent\data" -Destination ".\data" -Recurse -Force
```

Install Python and the separate Streamlit optional dependency:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,search,agent,ui]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m papertrail doctor
```

Then start the UI locally:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`. Stop with `Ctrl+C` in the PowerShell terminal.

### First offline run (no API key)

- Open the **Research** tab and select `2510.22344` if it appears in the indexed-paper list.
- Leave **Preview · no LLM**, **Baseline** and `top-k=3` selected.
- Ask `Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations.`
- Click **Run PaperTrail**. Inspect `Retrieved evidence` and `Execution diagnostics`. The UI will explicitly say **no answer generated**: this is expected and safe.
- In Preview, `llm_calls` should be `0`. There should be no arXiv or model API requests. Data/ and cached embedding models must already exist from earlier phases.

### Optional Live mode

Create `.env` in project root from `.env.example` and configure your model provider there (never in code, logs or GitHub). Choose **Live**, **Baseline**, check the explicit permission checkbox and click **Run PaperTrail**. The existing backend allows a bounded, single synthesis request. The UI doesn't provide an unrestricted internet-access mode for the research request. Your provider may charge for usage and receives bounded source excerpts. A review queue can be created only when generated claims were mechanically accepted and contain an in-graph Phase 4.3 audit.

### Optional new-paper import (network)

In **Paper library**, explicitly opt into network/model downloads, search public arXiv, review candidate metadata, and import one selected paper. This downloads source documents, computes embeddings locally (possibly downloading model weights), and generates an index in the local data directory. PDF readability and scientific correctness remain unverified. Large first-run embeddings may be slow on a CPU.

## Security and quality limits

- The UI contains **no automated semantic entailment classifier**. `quote_anchored` means literal text matching within registered source excerpts, not verification against an independent original PDF.
- The UI uses no network without explicit permission: arXiv search/import and Live model generation each have distinct buttons and consent.
- `focused`, `multi`, and `budgeted` are **exploratory Preview only**, consistent with Phase 4 CLI restrictions. Their four log-derived anchors are not gold relevance labels and do not establish generalization.
- The UI loads local files from `data/` only and only accepts valid arXiv identifiers. Human review files stay in `data/faithfulness_reviews/` (ignored by Git). Do not push `.env` or local user data.
- Output in the browser, including paper text, citations, and reviewer notes, may contain sensitive data; keep the local web server bound to 127.0.0.1.
- Demo/UI integration and operating-system-specific web serving still need verification on the user's Windows machine. In the development environment without Streamlit installed, only UI service unit tests and static Python validation are exercised.

## Quick troubleshooting

- **No papers found:** copy the previous `data/` folder from v1.5.5 into this project. `data/` is excluded from ZIP and Git.
- **Missing streamlit:** reinstall the optional `ui` extra: `python -m pip install -e ".[ui]"` using your active `.venv` Python.
- **Model weights unavailable in Preview:** use the local SentenceTransformer cache from an earlier run. Preview deliberately blocks first-time weight downloads.
- **Live disabled:** check `.env` and run `papertrail doctor`. Do not copy your key into chat.
- **Web port already in use:** use `--server.port 8502`, keep `--server.address 127.0.0.1`.
- **Review queue missing:** only a successfully synthesized Live answer can create one in the UI; the Phase 4.3 CLI can also create example review queues.

## Phase 6 compatibility note (v1.6.1)

Deprecated Streamlit `use_container_width` arguments were replaced by `width="stretch"`. The user verified the v1.6.0 Streamlit server started on Windows; the v1.6.1 browser UI still needs an acceptance check. See `docs/GITHUB_RELEASE.md`.

## Phase 6.1 progress update (v1.6.2)

Research and Paper Library import now display `st.progress` with a numeric percentage and the current **completed** workflow stage. In Research, the progress callback is driven by LangGraph state streaming and shows how many retrieval facets have finished. In import, percentages advance after PDF processing (25%), chunking (50%), embeddings (80%), indexing (95%) and final artifact validation (100%).

These are **milestone percentages**, not wall-clock estimates and not the internal `103/103` model-weight loading progress shown by `tqdm` in the terminal. A long embedding load can hold at one percentage. No timers/fake increments, additional LLM calls, or automatic network permissions are introduced. The CLI path is unchanged. If execution throws, the bar stays at the last confirmed milestone rather than jumping to 100%.

For acceptance on Windows, run the UI, choose an already indexed paper, Preview/Baseline/top-k=3, click **Run PaperTrail**, and verify the percentage plus facet labels update before the evidence appears. Also test the library import progress only when you explicitly intend to enable arXiv downloads.

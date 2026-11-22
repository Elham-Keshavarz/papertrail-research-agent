# Phase 3.1 — LangGraph Foundation (v1.0.0)

## Scope and honesty

PaperTrail now uses the **real LangGraph `StateGraph`** API for a local, deterministic skeleton. No API key is needed for this preview. The graph **does not yet call retrieval, discover papers, analyze papers, or generate answers.** The six plan steps are future tasks, not autonomous decisions or completed actions.

## Real graph topology

`START -> validate_request -> (invalid? finalize : plan_research) -> inspect_local_artifacts -> (ready | needs_paper | missing) -> finalize -> END`

Nodes are kept as pure callables where possible, to permit fast unit tests without external services:

| Node | Real behavior |
|---|---|
| `validate_request` | Validates question length, whitespace normalization and arXiv ID syntax |
| `plan_research` | Provides a **deterministic unexecuted outline**, not an LLM plan |
| `inspect_local_artifacts` | Checks existence of local chunks, vector matrix, embedding manifest, FAISS index, index manifest |
| `ready` | Records that all five files exist but are **unverified** |
| `needs_paper` | Requests selection of an arXiv paper ID in future phases |
| `missing` | Reports missing local artifacts; does not download or create them |
| `finalize` | Records that no answer was generated |

The state schema is `AgentState` (TypedDict) and trace uses LangGraph's additive `Annotated[..., operator.add]` reducer. `build_research_graph()` constructs and compiles the graph lazily. Each call to `run_agent_preview()` uses `graph.invoke(...)` with a bounded recursion limit. No checkpointer is used yet, and no state is persisted between runs.

## Installation on Windows

Extract the full ZIP. In the project root, copy the previous v0.9.0 `data/` folder (not `.venv`) if you want to verify presence of the existing 421-vector local index. Use these single-line commands separately:

```powershell
Copy-Item -Path "..\..\papertrail_phase2_7_retrieval_optimization_full\papertrail-research-agent\data" -Destination ".\data" -Recurse -Force
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,search,agent]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m papertrail doctor
.\.venv\Scripts\python.exe -m papertrail agent-preview --question "Which datasets were used to evaluate FAIR-RAG in the experiments?" --paper-id 2510.22344
```

If the `.venv` doesn't exist yet, run `py -3 -m venv .venv` first. The `agent` extra installs LangGraph; the `embeddings` and `search` extras preserve the previous research tools.

## Output contract

`status` is one of:

- `needs_paper_selection`: no paper ID supplied.
- `missing_local_artifacts`: the preview found one or more missing prerequisite local files.
- `local_artifacts_present_unverified`: all five expected files were found; **content integrity and correctness were not checked**.

`plan` contains six explicit `execution: not_executed` tasks. `trace` shows actual executed *graph nodes* rather than fictional LLM/tool events. The result always returns `llm_calls: 0`, `tool_calls: 0`, `research_executed: false` and `answer_generated: false`.

## What the tests establish

- Unit tests independently cover valid/invalid requests, safe IDs, deterministic outline, artifact checks, explicit status labels and no-output claims.
- A topology contract test inspects registered nodes, edges and conditional mapping without dependency installation.
- Native smoke tests, when LangGraph is installed (as in the recommended user command and CI), **compile and actually invoke the real graph**, including successful, missing-data, no-paper and invalid-input routes.
- All former retrieval, PDF, evaluation and optimization tests remain unchanged, except for the `doctor` milestone expectation.

## Security and quality limitations

- `inspect_local_artifacts` only calls `Path.is_file`. Even an empty/corrupt FAISS file would appear present in a preview; it is **not** a signal that search will succeed.
- Previously downloaded PDFs and their extracted source text are not included in ZIP releases and are untrusted input data.
- The extraction heuristic dropped roughly 30% of raw characters on the previously tested PDF; completeness and table fidelity need manual audit.
- Page-level relevance judgments are provisional. The tiny train/holdout split is not enough to claim generalization.
- Later phases will implement real tool adapters, multi-step planning, bounded retries, checkpointing and audited synthesis with citations.

## References

- [LangGraph Python package](https://pypi.org/project/langgraph/)
- [Official LangGraph documentation](https://docs.langchain.com/oss/python/learn)

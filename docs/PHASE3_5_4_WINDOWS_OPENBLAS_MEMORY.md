# Phase 3.5.4 — Windows OpenBLAS memory troubleshooting (`v1.4.4`)

## Reproduced user observation

On Windows, version 1.4.3 passed `python -m pytest -q`, but running `agent-answer`
for **both** `--mode preview` and `--mode live` printed repeated:

`OpenBLAS error: Memory allocation still failed after 10 retries, giving up.`

The log shows no JSON result and no evidence that a Live model call actually occurred.
Passing tests does not mean runtime model loading is guaranteed to work.

## Why this change is useful

OpenBLAS is a CPU math backend used by NumPy and other native libraries. Its
thread startup / workspace allocations can fail when a system is under memory
pressure or uses excessive parallelism. The primary fix is to constrain math
library threads **before import**, not after Python has imported NumPy/FAISS.
This version sets these environment variables to `1` at CLI bootstrap (not plain library import):
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`.
`TOKENIZERS_PARALLELISM` defaults to `false` unless explicitly set.

Both `python -m papertrail` and the `papertrail` console command invoke a
lightweight bootstrap **before** the CLI's native-library imports. Only commands
using these launchers apply the guard; importing PaperTrail into another
application does **not** alter the host application's thread settings. For custom deployments that require additional concurrency,
set `PAPERTRAIL_MATH_THREADS` to an integer from `1` to `8` **in the launching
environment**; `PAPERTRAIL_LIMIT_MATH_THREADS=0` fully disables the guard.
This default overrides inherited math-thread settings; set the explicit
PaperTrail override to opt into higher parallelism. Invalid thread counts
fall back safely to 1.

**Important:** `.env` is parsed too late to affect the early package bootstrap;
set these controls in the terminal before starting Python. Do not paste API
keys into troubleshooting reports.

## New full-project release (`v1.4.4`): installation steps

Extract the new `papertrail_phase3_5_4_windows_blas_memory_guard_full.zip` and open
its inner `papertrail-research-agent` directory. Copy previous local artifacts
without copying `.venv`; check that the example relative paths match your
Downloads folder layout.

```powershell
Copy-Item -Path "..\..\papertrail_phase3_5_3_registered_quote_selection_full\papertrail-research-agent\data" -Destination ".\data" -Recurse -Force
```

```powershell
Copy-Item -Path "..\..\papertrail_phase3_5_3_registered_quote_selection_full\papertrail-research-agent\.env" -Destination ".\.env" -Force
```

The second command is optional if you only plan to run Preview (no API key
needed), and must be performed locally; do not share `.env`.

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

Do not run `--mode live` unless Preview succeeds and you explicitly accept
one potentially billed LLM request. There is no automatic model retry.

## Windows PowerShell — run each command on its own line

If using the current 1.4.3 directory, you can try the no-install workaround:

```powershell
$env:OPENBLAS_NUM_THREADS="1"
```
```powershell
$env:OMP_NUM_THREADS="1"
```
```powershell
$env:MKL_NUM_THREADS="1"
```
```powershell
$env:NUMEXPR_NUM_THREADS="1"
```
```powershell
.\.venv\Scripts\python.exe -m papertrail agent-answer --paper-id 2510.22344 --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --top-k 3 --mode preview
```

In v1.4.4, the early guard is automatic; **test preview before Live**. This
preview sends **zero** answer-model API requests and needs no API key.

If the error persists even at a single thread, inspect free RAM and Windows
virtual-memory/pagefile settings, close other memory-heavy Python/VS Code
processes, and verify 64-bit Python. System/OS error causes cannot be
confirmed from a short OpenBLAS message alone. Avoid repeat Live calls until
Preview succeeds. If the local error is fixed, Live may still produce a
partial/unverified answer; see Phase 3.5.3 docs.

## Limitations and verification

- Unit/subprocess bootstrap tests verify environment variables are applied
  before CLI imports; they cannot simulate low-memory Windows behavior.
- The actual native Windows Preview/Live still needs confirmation.
- Thread capping **does not** alter citation checks or permit unsourced claims.
- A successful Preview is not evidence of correct scientific conclusions.
- Live mode requires an API key and might incur provider fees.

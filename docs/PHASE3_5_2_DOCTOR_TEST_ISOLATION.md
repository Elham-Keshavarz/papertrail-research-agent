# Phase 3.5.2 — Test isolation for `doctor` (v1.4.2)

## Symptom

After transferring a real `.env` from v1.4.0 to v1.4.1 and running the full test suite, `tests/test_cli.py::test_doctor_cli` failed because it asserted `live_research_available is False`. The code returned `True` when an API key was present and LangGraph was installed. The test was incorrectly dependent on the developer's local environment.

## Fix

- `test_doctor_cli_without_api_key` explicitly overrides the key with an empty environment value, so copied `.env` files cannot influence that test.
- `test_doctor_cli_with_api_key_env` uses a synthetic marker key and checks readiness is consistent with the installed LangGraph backend. It confirms no key leaks in output.
- `test_doctor_cli_with_dotenv_key` creates a throwaway `.env` containing a synthetic key in a temporary test directory. It exercises the exact reported regression and confirms no key leaks.
- The production readiness condition is intentionally unchanged: `live_research_available = langgraph_installed and api_key_configured`. It does not ping the provider. A configured key might be invalid, lack funds or use a mismatched model. Tests make zero live API calls.

## Run locally in PowerShell

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

```powershell
.\.venv\Scripts\python.exe -m papertrail doctor
```

```powershell
.\.venv\Scripts\python.exe -m papertrail agent-answer --paper-id 2510.22344 --question "Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations." --top-k 3 --mode preview
```

To check quote-validation changes with the real provider, run `--mode live` **only once deliberately after review**; it needs your own API key, may cost tokens, and you should share diagnostics without sharing `.env`.

The production synthesis logic and no-retry rule in v1.4.1 are unchanged by this patch.

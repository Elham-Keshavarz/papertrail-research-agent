# Phase 1 — How Paper Discovery works

## Goal

Take a short research keyword search, make one **read-only** arXiv request, safely parse the resulting Atom XML, and return validated metadata. This is a tool the later LangGraph agent can call. It is **not** an agent yet.

## 1. Domain model: `src/papertrail/models.py`

`PaperMetadata` represents **one paper**, independent of where it was found. Pydantic checks required fields and requires timezone-aware publication timestamps. Dates are represented as `datetime` in Python and ISO 8601 strings in JSON. `PaperSearchResult` contains all papers plus their query, start offset, and available total count.

The model uses `is_peer_reviewed=None` for "unknown". Even a journal reference isn't proof of peer review without independent confirmation.

## 2. HTTP adapter: `src/papertrail/discovery/arxiv_client.py`

`ArxivClient.search` accepts a string and a bounded result size. It creates the arXiv query parameters and requests `https://export.arxiv.org/api/query` over HTTPS. `httpx` handles HTTP and timeouts. The endpoint is fixed to avoid user-controlled destination URLs.

- **Timeout**: a request can't wait indefinitely.
- **Rate limiting**: at least three seconds between attempts, including retries. One client makes only sequential requests. Cross-process rate limiting is **not** implemented.
- **Retry**: 429 and selected transient server errors (5xx), or a transport failure, can retry up to the configured `max_retries`. Response parsing errors and non-transient HTTP errors don't retry.
- **Response size**: reject oversized metadata responses; full paper PDFs belong to Phase 2.
- **Resource lifecycle**: use `with ArxivClient(...) as client` to close its internal HTTP connection after use.

Client `max_results` is intentionally restricted to 1–20 for this phase, even though arXiv permits larger pages.

## 3. Parser: `src/papertrail/discovery/arxiv_parser.py`

`parse_arxiv_feed`:

1. Parses XML with `defusedxml` to reject dangerous XML features such as DTD entities.
2. Checks that the root is an Atom `feed` and detects error feeds.
3. Gets the total count (when present) from the OpenSearch extension.
4. For every Atom `entry`, reads its ID, title, authors, abstract, and publication/update dates.
5. Validates the arXiv identifier and constructs canonical `https://arxiv.org/abs/...` and `/pdf/...` URLs, instead of trusting link fields.
6. Converts the data into `PaperMetadata` objects and fails explicitly on invalid metadata.

An empty valid feed returns an empty list. A malformed entry raises an error; it never silently becomes a fake paper.

## 4. CLI: `src/papertrail/cli.py`

- `build_parser()` registers three commands and their command-line flags.
- `main()` loads `Settings`, creates the client, invokes `.search()` and JSON-serializes the Pydantic result.
- Exit code `0` means success (including 0 matches), `1` means an upstream/response error, `2` means invalid user input or configuration.

## 5. Tests

`tests/test_discovery_parser.py` uses a local XML fixture with a modern versioned ID and an old-style ID. It tests required fields, malicious XML, incorrect source domains, dates and invalid feeds.

`tests/test_discovery_client.py` uses `httpx.MockTransport`, a fake clock and an injected sleep function. This allows retry timing and rate limiting to be tested *without real sleeps or network calls*.

`tests/test_cli.py` also mocks the client to check actual JSON formatting and error exit codes. Real connectivity is a **separate manual smoke check**.

## Try it yourself (PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m papertrail search-papers --query "retrieval augmented generation" --limit 3
```

If the last command reports a network error, it doesn't imply the offline code tests failed. The API may be inaccessible from your network; check access and try again later, without rapid manual retries.

## What is explicitly out of scope?

- Downloading or storing PDFs
- Full-text search, chunking, embeddings, vector indexes
- LLM responses and final research reports
- Autonomous query rewriting, tool selection, LangGraph loops

These are implemented only in later phases.

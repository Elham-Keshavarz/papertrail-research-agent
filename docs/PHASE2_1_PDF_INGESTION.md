# Phase 2.1 — Download and extract arXiv PDFs

This phase is **document ingestion**, not yet RAG. There are no embeddings, no text chunks, no vector store, no LLM, and no agentic decisions.

## User journey

1. In Phase 1, search arXiv and choose an actual `paper_id` from returned metadata.
2. Run `python -m papertrail ingest-pdf --paper-id 2510.22344` from the project root.
3. The CLI validates the ID. No arbitrary URL is allowed.
4. If a cached PDF exists, use it; otherwise `ArxivPdfDownloader` fetches the file from an arXiv domain only.
5. Verify file size (20 MiB max) and PDF signature. Follow at most three **arXiv-only** PDF redirects; handle bounded retries on temporary network/HTTP problems. Requests are paced at least three seconds apart *per downloader instance*.
6. `extract_pdf` opens the bytes with PyMuPDF, rejects encrypted and corrupt PDFs, and extracts text page by page using `get_text('text', sort=True)`.
7. Results are saved locally at `data/papers/<id>.pdf` and `data/extracted/<id>.json`, using atomic file replacement.
8. The CLI prints a **summary only**, not the entire article, with `total_pages`, `extracted_pages`, `pages_without_text`, file paths, and `sha256`.

## Command

```powershell
.\.venv\Scripts\python.exe -m papertrail ingest-pdf --paper-id 2510.22344
```

Optional:

```powershell
.\.venv\Scripts\python.exe -m papertrail ingest-pdf --paper-id 2510.22344 --max-pages 25
.\.venv\Scripts\python.exe -m papertrail ingest-pdf --paper-id 2510.22344 --force
```

- `--max-pages 25` extracts only the first 25 pages. If more pages exist, `truncated` will be `true`.
- `--force` explicitly re-downloads even if a local PDF is cached.
- `PAPERTRAIL_MAX_PDF_PAGES` (default 100) and `PAPERTRAIL_DATA_DIR` (default `data`) can be set in `.env`.
- API keys are not needed for this phase. Network is required only on cache misses and forced downloads.

## Output model

Each JSON document records:

```json
{
  "paper_id": "2510.22344",
  "source_url": "https://arxiv.org/abs/2510.22344",
  "sha256": "...",
  "total_pages": 12,
  "extracted_pages": 12,
  "truncated": false,
  "empty_text_pages": [],
  "pages": [
    {
      "page_number": 1,
      "text": "The extracted text on page one ...",
      "character_count": 1234,
      "has_text": true
    }
  ]
}
```

**The example values above are illustrative, not a claim about the real paper.** Page numbers are **1-based** to match the PDF reader, not zero-based Python indices. Keeping the original arXiv ID, URL and page number is essential for later citations.

## Code map

| File | Responsibility |
|---|---|
| `ingestion/pdf_downloader.py` | Verify IDs, allowlisted HTTP fetches, redirects, retries, pacing, PDF magic and byte caps |
| `ingestion/pdf_extractor.py` | Open PDFs, extract sorted text per page and record blank pages |
| `ingestion/pipeline.py` | Coordinate cache, extraction and atomic persistence |
| `models.py` | Domain models from earlier phases remain intact |
| `cli.py` | Expose `ingest-pdf` and report a compact JSON summary |

## Known limits and safety

- `sort=True` improves approximate reading order; **it is not perfect** for tables, multiple columns, figures, and formulas.
- Scanned/image-only pages may have no extractable text. We record their page numbers; **OCR is not implemented**.
- PDFs are untrusted inputs; they may contain malicious instructions, so future agents must never interpret extracted text as commands.
- 20 MiB is the maximum downloaded PDF size, and no more than 200 pages can be extracted per run. Large/complex PDFs still require resource controls in a production deployment.
- The 3-second pacing is **per downloader instance**. Concurrent jobs must share global throttling to respect upstream policies.
- Full PDFs and extracted text go under `data/`, excluded from Git. **Do not commit or redistribute downloaded papers without checking their licences.**
- The current workflow is aimed at educational, small-scale, single-paper investigation; not bulk harvesting.
- SHA-256 lets us verify the cached bytes and connect derived text to an exact PDF version. It does not prove the content is true or peer reviewed.

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Tests build synthetic PDFs locally with PyMuPDF; no live arXiv call is required. Live ingestion must be tested on the user's own machine.

## Sources

- [PyMuPDF Page.get_text()](https://pymupdf.readthedocs.io/en/latest/page.html)
- [arXiv API Terms](https://info.arxiv.org/help/api/tou.html)
- [arXiv bulk access](https://info.arxiv.org/help/bulk_data.html)

# Phase 2.4.1 — Two-column PDF reading-order repair

## Why this patch exists

A real PaperTrail retrieval run on an academic PDF found excerpts where the left and right columns were mixed into one sentence. `page.get_text("text", sort=True)` sorts text geometrically, but that does **not** guarantee correct reading order for multi-column pages. The bad text harms both embedding quality and the trustworthiness of retrieved excerpts.

## Design (no new paid API and no new dependency)

1. Read geometric text blocks with `page.get_text("blocks", sort=False)`.
2. Conservatively recognize left and right columns when both contain substantial, vertically overlapping text.
3. Keep full-width title/heading blocks in place as section separators.
4. For each section, extract the **left** column from top to bottom, then the **right** column from top to bottom.
5. On pages that do not pass the conservative two-column check, use the existing `get_text("text", sort=True)` behavior.
6. Add `extraction_method` to each page (`two_column` or `simple`) so reviewers can see which method was used.

There is an optional CLI `--layout simple` mode for reproducible A/B comparisons. Default is `--layout auto`.

### Limitations

This is a **heuristic**, not a universal layout parser. Tables, sidebars, floating images, equations, footnotes, and unusual multi-column layouts may still be disordered. A clean extraction run is not proof of a faithful reading order. Inspect retrieved excerpts and page images before making claims, particularly for cited evidence. This patch does not perform OCR.

### Quality gate

- Unit tests use a generated PDF with a full-width heading, paired columns, and independent left/right paragraphs. The two-column extraction must keep the entirety of the left column before the right column, with no lost text.
- Existing extraction, chunking, embedding, and retrieval tests must continue to pass.
- On a real research PDF, visually compare the PDF to the extracted JSON on pages flagged `two_column`, and rerun the two previously problematic queries. Verify **coherent sentence order**, not just vector scores.
- Optionally compare the output from `--layout simple` to `--layout auto` for the same cached PDF. There is no automatic claim of retrieval improvement without manual or benchmark evaluation.

## Windows steps (full new ZIP)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,search]"
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m papertrail ingest-pdf --paper-id 2510.22344 --layout auto
.\.venv\Scripts\python.exe -m papertrail chunk-document --paper-id 2510.22344 --chunk-size 650 --overlap 100
.\.venv\Scripts\python.exe -m papertrail embed-document --paper-id 2510.22344 --batch-size 16
.\.venv\Scripts\python.exe -m papertrail index-document --paper-id 2510.22344
.\.venv\Scripts\python.exe -m papertrail search-chunks --paper-id 2510.22344 --query "What is Structured Evidence Assessment?" --top-k 3
.\.venv\Scripts\python.exe -m papertrail search-chunks --paper-id 2510.22344 --query "Which benchmarks evaluate FAIR-RAG?" --top-k 3
```

If you **copy the `data/` folder from an older version**, `ingest-pdf` will reuse the cached PDF but **overwrite extracted JSON**. Always rerun chunking, embeddings, and FAISS indexing after switching layout extraction, or you will be searching stale text. You can pass `--force` to embedding and indexing if needed. `--force` on `ingest-pdf` triggers a new PDF download, and is not necessary when the cached file is valid.

PDF text and derived artifacts are excluded from the ZIP / GitHub to avoid publishing third-party papers.

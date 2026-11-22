# Phase 2.2 — Deterministic Page-Aware Text Chunking

**Status:** Implemented offline, ready for live user verification. This is *not* semantic retrieval or a language-model agent yet.

## What problem does chunking solve?

A PDF is too large and noisy to feed directly to retrieval systems. Phase 2.1 generates JSON with the exact text and 1-based page number of each extracted page; Phase 2.2 splits this text into bounded pieces while retaining page provenance. A later embedding or lexical index will consume these chunks.

```text
arXiv PDF → [Phase 2.1] ExtractedDocument (pages)
         → [Phase 2.2] TextChunk (page + offsets + text)
         → data/chunks/<paper_id>.json
         → [Phase 2.3+] embeddings / retrieval (NOT implemented here)
```

## How to run (PowerShell)

Install the complete project from scratch, then run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m papertrail doctor
.\.venv\Scripts\python.exe -m papertrail ingest-pdf --paper-id 2510.22344
.\.venv\Scripts\python.exe -m papertrail chunk-document --paper-id 2510.22344
```

If upgrading in a **new folder**, `data/` from the previous folder is **not in the release ZIP**. Re-run `ingest-pdf` in the new folder (or explicitly copy your local `data/` directory). Chunking itself is offline and never contacts arXiv.

Optional parameters:

```powershell
.\.venv\Scripts\python.exe -m papertrail chunk-document --paper-id 2510.22344 --chunk-size 1500 --overlap 250
```

`chunk_size` is measured in **Python Unicode characters**, not tokens; valid range is 100–8000. `overlap` is an approximate *maximum* overlap of character ranges, and must be nonnegative and less than half the chunk size. Defaults: `1200` and `200`.

## Code files

- `src/papertrail/chunking/chunker.py`: pure-Python slicing algorithm and Pydantic output contracts.
- `src/papertrail/chunking/pipeline.py`: read extracted JSON, validate provenance, write chunks JSON.
- `src/papertrail/storage.py`: shared atomic write helper for both ingestion and chunking.
- `src/papertrail/cli.py`: register `chunk-document`, parse flags, summarize without dumping copyrighted text.
- `tests/test_chunker.py`: boundary, whitespace, Unicode, overlap, page integrity, deterministic IDs.
- `tests/test_chunk_pipeline.py`: file I/O, missing/corrupt data, old-style arXiv IDs and CLI errors.

## Why page-aware?

Each chunk belongs to exactly **one** page. It includes:

```json
{
  "chunk_id": "2510.22344@<source-digest>:p0001:000000-001200:<text-digest>",
  "paper_id": "2510.22344",
  "source_url": "https://arxiv.org/abs/2510.22344",
  "source_sha256": "<64-character PDF SHA-256>",
  "page_number": 1,
  "start_char": 0,
  "end_char": 1200,
  "text": "<the exact source slice, not paraphrased>"
}
```

This is **illustrative**, not an actual response. `end_char` is **exclusive**, and offsets are positions in the page's extracted `text` string. For all produced chunks:

```python
assert page.text[chunk.start_char:chunk.end_char] == chunk.text
```

IDs include the paper ID, a prefix of the input PDF's digest, source offsets, and a digest of the actual chunk content. They are reproducible for identical inputs and settings.

## Algorithm decisions

1. Process pages independently, in PDF order.
2. Prefer a paragraph break; fall back to newline, sentence end, space, then hard split.
3. Never exceed the selected character limit. Long unbroken words are split when necessary.
4. When the remaining text is shorter than the limit, emit the remainder rather than creating unnecessary chunks.
5. Rewind by `overlap` characters after each chunk (within the same page), with forward-progress guarantees and no dropped non-whitespace source characters.
6. Preserve exact text spans and one-based PDF page numbers. Do not normalize or reword chunk text.
7. Keep the processing deterministic and independent from API, embedding model or LLM.

Because boundary choice favors natural breaks, the actual overlap may be smaller than configured. Some chunks can begin in the middle of a sentence due to overlap. This is a **character-based baseline**, not semantic or token-aware chunking; comparing strategies is a future evaluation task.

## Safety and limitations

- The program validates a local extracted JSON and a matching arXiv ID before processing. The input file has a 48 MiB size ceiling, and outputs are written atomically.
- Unknown, mismatched or corrupted JSON files are rejected without modifying existing chunk outputs.
- Page citation is a reference to the *extracted text*, **not** a guarantee that PyMuPDF perfectly preserved columns, equations, figures or tables from the original PDF.
- All PDF content is untrusted data. Do not execute any text found inside it.
- `data/` is ignored by Git and not bundled in the release; PDF/article licensing remains separate from the project's code license.
- A chunk's existence does not imply that it is scientifically relevant or that the paper was peer reviewed.
- This phase does not measure retrieval quality. Next phase will add embeddings and evaluation baselines.

## Quality gates

- [x] Character and page bounds checked in offline tests
- [x] Exact offset matching; no cross-page chunks
- [x] Deterministic output and stable identifiers
- [x] Missing and corrupt extracted file fail gracefully
- [x] Full existing Phase 0–2.1 regression tests pass
- [ ] User has run the command on their locally ingested arXiv PDF
- [ ] User has inspected JSON for actual source page attribution

# Phase 2.3 — Local, traceable sentence embeddings

**Version 0.5.0.** This phase converts existing page-aware chunks into normalized, `float32` sentence embeddings. It does not implement FAISS or final answer generation, and it does not call a paid LLM API.

## 1. Why embeddings?

Character chunks are strings. A semantic retriever needs vectors so it can compare meaning even when a query and passage use different words. This phase builds the document vectors; query vectors and retrieval rankings come in Phase 2.4.

We use [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It produces **384-dimensional** vectors and is English-oriented. The weights are loaded locally after a one-time Hugging Face download. We pin the model revision to `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` to avoid changing weights/config on a moving `main` branch. The Apache-2.0 model license differs from the code's MIT license.

The model card warns that inputs longer than **256 word pieces are truncated**. The project uses the model's tokenizer to **count every input without truncation before any embedding**; if one is too long, it fails with a clear message rather than silently dropping evidence.

## 2. Quickstart (Windows PowerShell, in project root)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings]"
.\.venv\Scripts\python.exe -m papertrail doctor
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m papertrail ingest-pdf --paper-id 2510.22344
.\.venv\Scripts\python.exe -m papertrail chunk-document --paper-id 2510.22344 --chunk-size 650 --overlap 100
.\.venv\Scripts\python.exe -m papertrail embed-document --paper-id 2510.22344 --batch-size 16
```

**Why chunk-size 650 rather than 1200?** Earlier phases used character-based chunks of size 1200. Tokens are not equivalent to characters, so a 1200-character passage may exceed the model's token budget. The smaller size is a practical starting point, **not a guarantee**. If a PDF contains dense equations, numbers or symbols and even a 650-character chunk exceeds the token limit, try `--chunk-size 400 --overlap 70` and rerun `embed-document`. The PDF does not need to be downloaded again if `data/extracted/` exists.

The first model download requires internet and may pull substantial packages (PyTorch and Hugging Face). **No API key is required for a public model.** The program can use the local cache on later runs. If you have limited disk space, check it before installation.

The optional `[embeddings]` extra is intentional: `pip install -e ".[dev]"` keeps CI and earlier-phase tests lightweight; the above command installs the model library as well.

## 3. Output files

```text
data/
  chunks/2510.22344.json
  embeddings/2510.22344.npy
  embeddings/2510.22344.manifest.json
```

- `*.npy`: `float32` matrix with one row per chunk, all rows L2-normalized (ready for cosine/dot-product retrieval).
- `*.manifest.json`: schema version, source PDF hash, exact chunk file hash, ordered chunk IDs, pinned model name/revision, row count, dimension, token limits and matrix SHA-256.
- **Row i** in `*.npy` corresponds to `chunk_ids[i]` in the manifest and the same `chunk_id` in `data/chunks/`.

On first success the CLI prints `status: embedded`; on a repeat with matching source, model and verified vectors, it prints `status: cached`. Use `--force` to deliberately rebuild.

## 4. Code walkthrough

| Module | Responsibility |
|---|---|
| `embedding/encoder.py` | Load pinned SentenceTransformer lazily; exact token counting, batch encoding |
| `embedding/pipeline.py` | Read and validate chunks; check all token lengths; build normalized matrix; cache with hashes |
| `embedding/__init__.py` | Public entry points for embedding step |
| `cli.py` | Handle `embed-document` flags and render a metadata-only JSON summary |
| `tests/test_embedding_pipeline.py` | Test success, cache, corruption, token limit, data mismatch and end-to-end fixture |
| `tests/test_embedding_cli.py` | Test error handling and output privacy |

`Encoder` is a small Python `Protocol`, enabling a fake encoder in tests. The tests **do not download the actual ML model**, so a passing test suite does not by itself prove the real model can download and run on your machine. The final `embed-document` command is the separate live smoke test.

## 5. How cache safety works

A cache is reused only when the chunk JSON hash, ordered chunk IDs, PDF hash, model name and pinned revision match. The matrix itself is checked with SHA-256, shape, `float32`, finiteness and unit-length norms. Matrix and manifest are saved via temporary files and atomic rename; the manifest is written last. If an operation fails or a cache file becomes corrupted, the next run rebuilds it.

NumPy `.npy` files are loaded with `allow_pickle=False`, avoiding Python object deserialization. Do not manually edit manifest hashes to force reuse: that bypasses the intended provenance contract.

## 6. Known limitations / quality checklist

- The text extraction and character-based chunk boundaries may still be imperfect for tables, math or two-column layouts. This phase only protects against model *input truncation*, not source-PDF extraction mistakes.
- Chunks must be short enough for the model's tokenizer. Some shorter but token-dense passages may need further rechunking.
- The current model is a lightweight, English-language **baseline**, not necessarily the best scientific retrieval encoder. Phase 2.6 will compare models and chunk strategies on labelled questions before upgrading.
- The project tests formatting, provenance, input validation, cache behaviour and vector invariants. **Retrieval relevance is not yet measured**.
- The GitHub repository must not contain extracted full papers, cached embeddings or any API secrets; `data/` is gitignored.
- Different library/hardware versions can cause small numerical differences. The model revision is pinned, but third-party Python dependency versions are range-bounded rather than fully locked; add a lock file before production deployment.

## 7. Troubleshooting

- `Run chunk-document first`: no chunk JSON in the **current** project folder; re-run `ingest-pdf`, then `chunk-document`.
- `Embedding dependencies missing`: run `pip install -e ".[dev,embeddings]"` rather than just `[dev]`.
- `Could not load local embedding model`: first download needs a working connection to Hugging Face; check your connection, available disk space and any proxy settings.
- `chunk(s) exceed the model's ... token limit`: re-chunk using `--chunk-size 400 --overlap 70`, then retry.
- `status: cached`: the existing matrix passed integrity checks and the model does not need to be loaded again.

Official references: [Sentence Transformers](https://www.sbert.net/docs/sentence_transformer/usage/usage.html), [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [NumPy load security](https://numpy.org/doc/stable/reference/generated/numpy.load.html).

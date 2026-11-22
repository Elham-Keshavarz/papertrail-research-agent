# Phase 2.5 — Hybrid Retrieval (PaperTrail v0.7.1)

This milestone adds **local** lexical retrieval, reciprocal rank fusion and neighbor evidence expansion to the existing per-paper FAISS/MiniLM pipeline. There is **no LLM synthesis or answer-generation endpoint**.

## Run on Windows PowerShell

Unzip the complete project and open `papertrail-research-agent` in VS Code. To avoid downloading/encoding the same article again, copy your validated `data` directory from v0.6.1 into this project's root. Do not copy `.venv`; create a fresh one.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,search]"
.\.venv\Scripts\python.exe -m papertrail doctor
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m papertrail hybrid-search --paper-id 2510.22344 --query "Which datasets were used to evaluate FAIR-RAG in the experiments?" --top-k 10 --context-window 1
.\.venv\Scripts\python.exe -m papertrail evaluate-benchmarks --paper-id 2510.22344 --top-k 10 --context-window 1
```

**Copy cached source data:** From inside the new project folder, use PowerShell `Copy-Item -Recurse -Force "<old-project-folder>\data" .\data`, or copy using File Explorer. Make sure the new folder contains `data/chunks/2510.22344.json`, `data/embeddings/2510.22344.npy`, its manifest, and optionally the cached `.faiss` index. If the index file is missing it is automatically rebuilt. If the embedding files are missing, re-run ingestion, chunking (`--chunk-size 400 --overlap 60`) and embedding (`--batch-size 16`) in that order.

## Methods and data flow

```text
Original extracted article -> page-aware chunks -> pinned normalized MiniLM vectors
                                                    |               |
User question ----> MiniLM query embedding -> FAISS IndexFlatIP (dense top 20)
           |                                        |
           +----> tokenizer -> in-memory BM25 -------+---> RRF(k=60) ----> top-k
                                         (sparse top 60)             |
                                                                  source-cited neighbors
```

1. **Dense search** reuses the existing verified FAISS index and MiniLM revision. The dense score is cosine similarity, **not probability**.
2. **BM25** indexes the exact same chunk data and uses a deterministic Unicode tokenizer. Nonnegative BM25 scores are **not probabilities**. BM25 is built in memory for each query; this is fine for hundreds of chunks but is not a high-volume search service.
3. **Reciprocal Rank Fusion (RRF)** uses `score = 1/(60 + dense_rank) + 1/(60 + bm25_rank)`; missing ranks contribute zero. Do **not** combine raw BM25 and cosine values. RRF is a ranking score, not evidence sufficiency.
4. **Evidence expansion** adds `context` containing up to `--context-window` original chunks on each side of each primary hit **only within the same page**. Metadata, text, offsets, and page citations are preserved per chunk; contexts are not fabricated or altered. Overlap or duplicate context across hits may occur and is not hidden.
5. Each result reports `dense_rank`, `bm25_rank`, `dense_score`, `bm25_score`, `score` (RRF), and `citation` with article/page. Null rank means the chunk was not retrieved by that method within its candidate window.

The original `search-chunks` command is unchanged. Use it to compare FAISS-only output against `hybrid-search` on the same frozen corpus and query.

## Quality check: explicit benchmark-name coverage

`evaluate-benchmarks` always uses the **same** question for both methods:

> Which datasets were used to evaluate FAIR-RAG in the experiments?

The four expected names were verified in the previously extracted article by the user: `HotpotQA`, `2WikiMultiHopQA`, `MuSiQue`, `TriviaQA`.

It also scans the full verified chunk corpus so you can distinguish missing source names from missed retrieval. It measures which strings **actually occur in the cited retrieved passages** and reports `found`, `missing`, `covered_count`, `expected_count`, `recall_by_name`, `cited_pages`. Dense-only uses its `top_k` main hits. Hybrid uses its `top_k` primary hits **and** their separately cited neighbors. This comparison deliberately includes the benefit of neighbor expansion, so it is **not** a controlled BM25-vs-FAISS component ablation.

**Important limitations:**

- A higher `recall_by_name` is *not* proof of better scientific QA, relevance, completeness, source accuracy or hallucination rate. It is a narrow, string-level, single-question diagnostic.
- The quality of PDF extraction remains an open issue: the v0.6.1 `auto` extractor kept about 125,031 vs ~178,156 characters previously, and reported all 30 pages as two-column. Manually inspect full-text loss and tables before relying on results for high-stakes research.
- Short or badly fragmented PDF text, hyphenation, tables and equations remain imperfect. Neighbor chunks can still contain broken sentences. The algorithm does not guarantee every required source is found.
- A single paper is searched at a time; cross-paper retrieval, agents, reranking, LLM synthesis and verified answer writing are future milestones.

## Verification gates

- Unit and integration tests run without a network and use a fake embedding backend and FAISS adapter.
- A **native FAISS smoke test** runs on installations with `faiss-cpu`; it is skipped in environments lacking that optional library.
- On a real environment, compare top results and manual page citations. Do not declare coverage improvement until the actual printed results confirm it.
- Files are loaded through SHA-256 and model-revision-checked bundle and index readers. Cached `.faiss` files are local trusted artifacts, **not safe to deserialize from untrusted uploads**.
- The source `data/` folder is git-ignored and excluded from the ZIP; do not publish article PDFs indiscriminately.

## v0.7.1 test isolation fix

The `test_embedding_cli_missing_input_error` case now executes from a pytest temporary working directory. This ensures that a user-provided `data/` folder and cached embeddings cannot change the expected missing-input result. This is a test-only fix; no retrieval, model, indexing or data format changes.

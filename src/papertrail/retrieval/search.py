"""Query embedding + exact FAISS cosine search + page-cited evidence hits."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from papertrail.embedding.encoder import Encoder, SentenceTransformerEncoder
from papertrail.embedding.pipeline import load_saved_embedding_bundle
from papertrail.ingestion.pdf_downloader import validate_paper_id
from .faiss_store import RetrievalError, load_paper_index

MAX_QUERY_CHARACTERS = 2000
MAX_TOP_K = 20


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rank: int = Field(ge=1)
    score: float = Field(ge=-1.0001, le=1.0001)
    paper_id: str
    source_url: str
    page_number: int = Field(ge=1)
    chunk_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str
    citation: str


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    query: str
    model_name: str
    model_revision: str
    vector_count: int = Field(gt=0)
    top_k: int = Field(ge=1)
    index_cached: bool
    hits: list[SearchHit]


def search_paper(
    paper_id: str, query: str, *, data_dir: Path, top_k: int = 5,
    encoder_factory: Callable[[str, str], Encoder] | None = None,
    faiss_backend: Any | None = None,
) -> SearchResult:
    """Return grounded retrieval results; scores are cosine similarity, not probability."""
    paper_id = validate_paper_id(paper_id)
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARACTERS:
        raise RetrievalError(f"query must contain 1 to {MAX_QUERY_CHARACTERS} characters")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise RetrievalError(f"top_k must be an integer from 1 to {MAX_TOP_K}")
    try:
        bundle = load_saved_embedding_bundle(paper_id, data_dir=data_dir)
    except (OSError, ValueError) as exc:
        raise RetrievalError(str(exc)) from exc
    index, index_report = load_paper_index(
        paper_id, data_dir=data_dir, bundle=bundle, faiss_backend=faiss_backend
    )
    manifest = bundle.manifest
    query = query.strip()
    try:
        encoder = (encoder_factory or SentenceTransformerEncoder)(manifest.model_name, manifest.model_revision)
        if encoder.dimension != manifest.embedding_dimension:
            raise RetrievalError("Query encoder dimension does not match saved document vectors")
        length = encoder.token_count(query)
        if not isinstance(length, int) or length <= 0 or length > encoder.max_sequence_length:
            raise RetrievalError("Query exceeds model token limit; shorten the query")
        if not hasattr(encoder, "encode_query"):
            raise RetrievalError("Query encoder does not implement encode_query")
        vector = np.asarray(encoder.encode_query(query), dtype=np.float32)
        if vector.shape != (manifest.embedding_dimension,) or not np.isfinite(vector).all():
            raise RetrievalError("Query embedding has invalid dimensions or values")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm < 1e-12:
            raise RetrievalError("Query embedding has zero norm")
        query_vectors = np.ascontiguousarray((vector / norm).reshape(1, -1), dtype=np.float32)
        scores, positions = index.search(query_vectors, min(top_k, manifest.vector_count))
        if scores.shape != positions.shape or scores.shape != (1, min(top_k, manifest.vector_count)):
            raise RetrievalError("FAISS returned unexpected search dimensions")
        hits: list[SearchHit] = []
        seen: set[int] = set()
        for rank, (position, score) in enumerate(zip(positions[0], scores[0]), 1):
            row = int(position)
            value = float(score)
            if row < 0 or row >= len(bundle.document.chunks) or row in seen:
                raise RetrievalError("FAISS returned an invalid or duplicate chunk position")
            if not np.isfinite(value) or not -1.0001 <= value <= 1.0001:
                raise RetrievalError("FAISS returned an invalid cosine score")
            seen.add(row)
            chunk = bundle.document.chunks[row]
            hits.append(SearchHit(
                rank=rank, score=round(value, 6), paper_id=chunk.paper_id,
                source_url=chunk.source_url, page_number=chunk.page_number,
                chunk_id=chunk.chunk_id, start_char=chunk.start_char,
                end_char=chunk.end_char, text=chunk.text,
                citation=f"{chunk.source_url}#page={chunk.page_number}",
            ))
        return SearchResult(
            paper_id=paper_id, query=query, model_name=manifest.model_name,
            model_revision=manifest.model_revision, vector_count=manifest.vector_count,
            top_k=top_k, index_cached=index_report.cached, hits=hits,
        )
    except RetrievalError:
        raise
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        raise RetrievalError(f"Could not search vectors: {type(exc).__name__}") from exc

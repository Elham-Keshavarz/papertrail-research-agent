"""Create, validate, and cache page-attributed embedding matrices for a paper."""
from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from papertrail.chunking.chunker import ChunkedDocument
from papertrail.ingestion.pdf_downloader import validate_paper_id
from papertrail.storage import atomic_write_bytes
from .encoder import Encoder, MODEL_NAME, MODEL_REVISION, SentenceTransformerEncoder

MAX_CHUNK_FILE_BYTES = 64 * 1024 * 1024
MAX_VECTOR_FILE_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class EmbeddingError(ValueError):
    """Invalid data, unsafe model input, or failed embedding step."""


class EmbeddingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    paper_id: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chunk_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_name: str
    model_revision: str
    normalized: bool
    embedding_dimension: int = Field(gt=0)
    max_sequence_length: int = Field(gt=0)
    min_input_tokens: int = Field(ge=0)
    max_input_tokens: int = Field(ge=0)
    vector_count: int = Field(ge=0)
    vector_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chunk_ids: list[str]


@dataclass(frozen=True)
class EmbeddingRun:
    manifest: EmbeddingManifest
    vectors_path: Path
    manifest_path: Path
    cached: bool


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_chunks(document: ChunkedDocument, paper_id: str) -> None:
    if document.paper_id != paper_id:
        raise EmbeddingError("Chunk paper ID does not match requested paper ID")
    if document.total_chunks != len(document.chunks) or not document.chunks:
        raise EmbeddingError("Chunk document must contain a nonempty, consistent chunk list")
    if document.source_url != f"https://arxiv.org/abs/{paper_id}":
        raise EmbeddingError("Chunk document source URL does not match paper ID")
    if len({c.chunk_id for c in document.chunks}) != len(document.chunks):
        raise EmbeddingError("Duplicate chunk IDs are not allowed")
    for chunk in document.chunks:
        if (chunk.paper_id != paper_id or chunk.source_url != document.source_url or
            chunk.source_sha256 != document.source_sha256 or
            chunk.page_number > document.extracted_pages or
            chunk.end_char <= chunk.start_char or
            chunk.end_char - chunk.start_char != len(chunk.text) or
            not chunk.text.strip()):
            raise EmbeddingError(f"Inconsistent chunk metadata: {chunk.chunk_id}")


def _valid_matrix(vectors: np.ndarray, *, count: int, dimension: int) -> bool:
    if vectors.dtype != np.float32 or vectors.shape != (count, dimension) or not np.isfinite(vectors).all():
        return False
    norms = np.linalg.norm(vectors, axis=1)
    return bool(np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3))


def _try_cache(
    manifest_path: Path, vectors_path: Path, *, paper_id: str, chunk_file_sha256: str,
    source_sha256: str, chunk_ids: list[str], model_name: str, model_revision: str,
) -> EmbeddingManifest | None:
    if not manifest_path.is_file() or not vectors_path.is_file():
        return None
    try:
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES or vectors_path.stat().st_size > MAX_VECTOR_FILE_BYTES:
            return None
        manifest = EmbeddingManifest.model_validate_json(manifest_path.read_bytes())
        if not (
            manifest.paper_id == paper_id and manifest.chunk_file_sha256 == chunk_file_sha256
            and manifest.source_sha256 == source_sha256 and manifest.chunk_ids == chunk_ids
            and manifest.model_name == model_name and manifest.model_revision == model_revision
            and manifest.normalized and manifest.vector_count == len(chunk_ids)
        ):
            return None
        raw = vectors_path.read_bytes()
        if _digest(raw) != manifest.vector_file_sha256:
            return None
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        if not isinstance(arr, np.ndarray) or not _valid_matrix(arr, count=len(chunk_ids), dimension=manifest.embedding_dimension):
            return None
        return manifest
    except (OSError, ValueError, ValidationError, EOFError, KeyError, TypeError):
        return None


def embed_saved_document(
    paper_id: str,
    *,
    data_dir: Path,
    batch_size: int = 16,
    force: bool = False,
    encoder_factory: Callable[[str, str], Encoder] | None = None,
    model_name: str = MODEL_NAME,
    model_revision: str = MODEL_REVISION,
) -> EmbeddingRun:
    """Embed an existing chunk JSON using the pinned local model, then cache float32 .npy.

    The manifest is the last file written. Cached vectors are hash-verified before use.
    A model-token preflight prevents silent loss of text due to truncation.
    """
    paper_id = validate_paper_id(paper_id)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 128:
        raise EmbeddingError("batch_size must be an integer between 1 and 128")
    if not model_name or not model_revision:
        raise EmbeddingError("model_name and model_revision are required")
    stem = paper_id.replace("/", "__")
    source_path = data_dir / "chunks" / f"{stem}.json"
    vectors_path = data_dir / "embeddings" / f"{stem}.npy"
    manifest_path = data_dir / "embeddings" / f"{stem}.manifest.json"
    if not source_path.is_file():
        raise EmbeddingError(f"Chunk JSON not found: {source_path}. Run chunk-document first")
    try:
        if source_path.stat().st_size > MAX_CHUNK_FILE_BYTES:
            raise EmbeddingError("Chunk JSON is too large")
        raw = source_path.read_bytes()
        document = ChunkedDocument.model_validate_json(raw)
        _validate_chunks(document, paper_id)
    except EmbeddingError:
        raise
    except (OSError, ValidationError, ValueError) as exc:
        raise EmbeddingError("Could not read or validate chunk JSON") from exc

    source_hash = _digest(raw)
    chunk_ids = [c.chunk_id for c in document.chunks]
    if not force:
        manifest = _try_cache(
            manifest_path, vectors_path, paper_id=paper_id, chunk_file_sha256=source_hash,
            source_sha256=document.source_sha256, chunk_ids=chunk_ids,
            model_name=model_name, model_revision=model_revision,
        )
        if manifest is not None:
            return EmbeddingRun(manifest, vectors_path, manifest_path, cached=True)

    try:
        encoder = (encoder_factory or SentenceTransformerEncoder)(model_name, model_revision)
        if not 1 <= encoder.dimension <= 16_384 or not 1 <= encoder.max_sequence_length <= 131_072:
            raise EmbeddingError("Model reported invalid vector dimension or token limit")
        lengths = [encoder.token_count(c.text) for c in document.chunks]
        if not all(isinstance(t, int) and 0 < t <= 1_000_000 for t in lengths):
            raise EmbeddingError("Model tokenizer returned invalid token counts")
        over = [(i, length) for i, length in enumerate(lengths) if length > encoder.max_sequence_length]
        if over:
            index, token_count = over[0]
            raise EmbeddingError(
                f"{len(over)} chunk(s) exceed the model's {encoder.max_sequence_length}-token limit. "
                f"First: chunk {index + 1} ({token_count} tokens), page {document.chunks[index].page_number}. "
                "No text was silently truncated. Re-run: papertrail chunk-document "
                f"--paper-id {paper_id} --chunk-size 650 --overlap 100; then embed again. "
                "If the warning remains, reduce --chunk-size further."
            )
        # No network/ML call happens until every input passes the truncation check.
        vectors = np.asarray(
            encoder.encode_documents([c.text for c in document.chunks], batch_size=batch_size), dtype=np.float32
        )
        if vectors.shape != (len(document.chunks), encoder.dimension) or not np.isfinite(vectors).all():
            raise EmbeddingError("Embedding result has invalid shape or nonfinite values")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
            raise EmbeddingError("Embedding model returned a zero or invalid vector")
        vectors = np.ascontiguousarray(vectors / norms[:, None], dtype=np.float32)
        if not _valid_matrix(vectors, count=len(document.chunks), dimension=encoder.dimension):
            raise EmbeddingError("Normalized embedding matrix failed validation")
        buffer = io.BytesIO()
        np.save(buffer, vectors, allow_pickle=False)
        blob = buffer.getvalue()
        if len(blob) > MAX_VECTOR_FILE_BYTES:
            raise EmbeddingError("Embedding matrix exceeds maximum file size")
        manifest = EmbeddingManifest(
            paper_id=paper_id, source_sha256=document.source_sha256,
            chunk_file_sha256=source_hash, model_name=model_name, model_revision=model_revision,
            normalized=True, embedding_dimension=encoder.dimension,
            max_sequence_length=encoder.max_sequence_length,
            min_input_tokens=min(lengths), max_input_tokens=max(lengths),
            vector_count=len(document.chunks), vector_file_sha256=_digest(blob), chunk_ids=chunk_ids,
        )
        # Write matrix first and manifest last: stale/incomplete pairs are rejected on next run.
        atomic_write_bytes(vectors_path, blob)
        atomic_write_bytes(manifest_path, manifest.model_dump_json(indent=2).encode("utf-8"))
        return EmbeddingRun(manifest, vectors_path, manifest_path, cached=False)
    except EmbeddingError:
        raise
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        raise EmbeddingError(f"Unable to create embeddings: {exc}") from exc


@dataclass(frozen=True)
class SavedEmbeddingBundle:
    """Fully verified chunks, manifest and normalized vectors, in matching row order."""

    document: ChunkedDocument
    manifest: EmbeddingManifest
    vectors: np.ndarray
    chunk_file_sha256: str


def load_saved_embedding_bundle(paper_id: str, *, data_dir: Path) -> SavedEmbeddingBundle:
    """Load saved embeddings for retrieval; reject stale or tampered artifacts.

    Unlike the embedding cache, stale inputs are a hard error here: retrieval
    must never silently cite chunks that do not match its vectors.
    """
    paper_id = validate_paper_id(paper_id)
    stem = paper_id.replace("/", "__")
    chunk_path = data_dir / "chunks" / f"{stem}.json"
    manifest_path = data_dir / "embeddings" / f"{stem}.manifest.json"
    vectors_path = data_dir / "embeddings" / f"{stem}.npy"
    for path, limit in ((chunk_path, MAX_CHUNK_FILE_BYTES),
                        (manifest_path, MAX_MANIFEST_BYTES),
                        (vectors_path, MAX_VECTOR_FILE_BYTES)):
        if not path.is_file():
            raise EmbeddingError(f"Required file not found: {path}. Run chunk-document and embed-document first")
        if path.stat().st_size > limit:
            raise EmbeddingError(f"Saved artifact exceeds size limit: {path.name}")
    try:
        chunk_bytes = chunk_path.read_bytes()
        doc = ChunkedDocument.model_validate_json(chunk_bytes)
        _validate_chunks(doc, paper_id)
        manifest = EmbeddingManifest.model_validate_json(manifest_path.read_bytes())
        vector_bytes = vectors_path.read_bytes()
        if not (
            manifest.paper_id == paper_id
            and manifest.source_sha256 == doc.source_sha256
            and manifest.chunk_file_sha256 == _digest(chunk_bytes)
            and manifest.vector_file_sha256 == _digest(vector_bytes)
            and manifest.normalized
            and manifest.vector_count == doc.total_chunks
            and manifest.chunk_ids == [c.chunk_id for c in doc.chunks]
        ):
            raise EmbeddingError("Saved embeddings are stale or inconsistent. Run embed-document again")
        vectors = np.load(io.BytesIO(vector_bytes), allow_pickle=False)
        if not isinstance(vectors, np.ndarray) or not _valid_matrix(
            vectors, count=manifest.vector_count, dimension=manifest.embedding_dimension
        ):
            raise EmbeddingError("Saved vectors are invalid or not normalized")
        return SavedEmbeddingBundle(doc, manifest, np.ascontiguousarray(vectors), _digest(chunk_bytes))
    except EmbeddingError:
        raise
    except (OSError, ValueError, ValidationError, EOFError, KeyError, TypeError) as exc:
        raise EmbeddingError("Cannot validate saved embedding artifacts") from exc

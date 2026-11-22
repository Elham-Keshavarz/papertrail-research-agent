"""An exact per-paper FAISS cosine-search index with verified source attribution.

All files are local and are hashed, but a checksum is NOT a digital signature.
Never load index data supplied by untrusted users: FAISS deserialization is native code.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from papertrail.embedding.pipeline import SavedEmbeddingBundle, load_saved_embedding_bundle
from papertrail.ingestion.pdf_downloader import validate_paper_id
from papertrail.storage import atomic_write_bytes

MAX_INDEX_BYTES = 128 * 1024 * 1024
MAX_INDEX_MANIFEST_BYTES = 2 * 1024 * 1024


class RetrievalError(ValueError):
    """A search input, index, or supporting source is invalid."""


class IndexManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    paper_id: str
    model_name: str
    model_revision: str
    vector_count: int = Field(gt=0)
    dimension: int = Field(gt=0)
    metric: str = Field(pattern=r"^inner_product$")
    index_type: str = Field(pattern=r"^IndexFlatIP$")
    vector_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chunk_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    index_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class IndexBuild:
    manifest: IndexManifest
    index_path: Path
    manifest_path: Path
    cached: bool


def _digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _backend(override: Any | None) -> Any:
    if override is not None:
        return override
    try:
        import faiss
    except ImportError as exc:
        raise RetrievalError('FAISS is not installed. Run: pip install -e ".[dev,embeddings,search]"') from exc
    return faiss


def _paths(data_dir: Path, paper_id: str) -> tuple[Path, Path]:
    stem = validate_paper_id(paper_id).replace("/", "__")
    folder = data_dir / "indexes"
    return folder / f"{stem}.faiss", folder / f"{stem}.manifest.json"


def _manifest_matches(manifest: IndexManifest, bundle: SavedEmbeddingBundle, paper_id: str) -> bool:
    m = bundle.manifest
    return bool(
        manifest.paper_id == paper_id
        and manifest.model_name == m.model_name
        and manifest.model_revision == m.model_revision
        and manifest.vector_count == m.vector_count
        and manifest.dimension == m.embedding_dimension
        and manifest.metric == "inner_product"
        and manifest.index_type == "IndexFlatIP"
        and manifest.vector_file_sha256 == m.vector_file_sha256
        and manifest.chunk_file_sha256 == bundle.chunk_file_sha256
    )


def _cached_index(index_path: Path, manifest_path: Path, bundle: SavedEmbeddingBundle,
                  paper_id: str) -> tuple[IndexManifest, bytes] | None:
    if not index_path.is_file() or not manifest_path.is_file():
        return None
    try:
        if index_path.stat().st_size > MAX_INDEX_BYTES or manifest_path.stat().st_size > MAX_INDEX_MANIFEST_BYTES:
            return None
        manifest = IndexManifest.model_validate_json(manifest_path.read_bytes())
        if not _manifest_matches(manifest, bundle, paper_id):
            return None
        blob = index_path.read_bytes()
        if not blob or _digest(blob) != manifest.index_file_sha256:
            return None
        return manifest, blob
    except (OSError, ValidationError, ValueError):
        return None


def _build(paper_id: str, data_dir: Path, bundle: SavedEmbeddingBundle,
           backend: Any, force: bool) -> IndexBuild:
    index_path, manifest_path = _paths(data_dir, paper_id)
    if not force:
        cached = _cached_index(index_path, manifest_path, bundle, paper_id)
        if cached:
            return IndexBuild(cached[0], index_path, manifest_path, True)
    try:
        index = backend.IndexFlatIP(bundle.manifest.embedding_dimension)
        index.add(np.ascontiguousarray(bundle.vectors, dtype=np.float32))
        if index.ntotal != bundle.manifest.vector_count:
            raise RetrievalError("FAISS indexed fewer vectors than expected")
        blob = np.asarray(backend.serialize_index(index), dtype=np.uint8).tobytes()
        if not blob or len(blob) > MAX_INDEX_BYTES:
            raise RetrievalError("Invalid FAISS serialization size")
        manifest = IndexManifest(
            paper_id=paper_id, model_name=bundle.manifest.model_name,
            model_revision=bundle.manifest.model_revision,
            vector_count=bundle.manifest.vector_count,
            dimension=bundle.manifest.embedding_dimension,
            metric="inner_product", index_type="IndexFlatIP",
            vector_file_sha256=bundle.manifest.vector_file_sha256,
            chunk_file_sha256=bundle.chunk_file_sha256,
            index_file_sha256=_digest(blob),
        )
        # Commit data before metadata; a torn write never passes _cached_index.
        atomic_write_bytes(index_path, blob)
        atomic_write_bytes(manifest_path, manifest.model_dump_json(indent=2).encode("utf-8"))
        return IndexBuild(manifest, index_path, manifest_path, False)
    except RetrievalError:
        raise
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        raise RetrievalError(f"Unable to build FAISS index: {type(exc).__name__}") from exc


def build_paper_index(paper_id: str, *, data_dir: Path, force: bool = False,
                      faiss_backend: Any | None = None) -> IndexBuild:
    """Persist an exact cosine index for already-generated document embeddings."""
    paper_id = validate_paper_id(paper_id)
    try:
        bundle = load_saved_embedding_bundle(paper_id, data_dir=data_dir)
    except (ValueError, OSError) as exc:
        raise RetrievalError(str(exc)) from exc
    return _build(paper_id, data_dir, bundle, _backend(faiss_backend), force)


def load_paper_index(paper_id: str, *, data_dir: Path, bundle: SavedEmbeddingBundle,
                     faiss_backend: Any | None = None, force: bool = False) -> tuple[Any, IndexBuild]:
    """Load a hash-verified index created locally; rebuild an invalid cache once."""
    backend = _backend(faiss_backend)
    report = _build(paper_id, data_dir, bundle, backend, force)
    cached = _cached_index(report.index_path, report.manifest_path, bundle, paper_id)
    if cached is None:
        raise RetrievalError("Index validation failed after build")
    try:
        # Local integrity check protects against accidental corruption, not malicious files.
        index = backend.deserialize_index(np.frombuffer(cached[1], dtype=np.uint8).copy())
        if (
            index.d != bundle.manifest.embedding_dimension
            or index.ntotal != bundle.manifest.vector_count
            or index.metric_type != backend.METRIC_INNER_PRODUCT
        ):
            raise RetrievalError("Cached index dimensions, count or metric mismatch. Rebuild with --force")
    except RetrievalError:
        raise
    except (RuntimeError, ValueError, TypeError) as exc:
        raise RetrievalError("Could not load FAISS index; rerun index-document --force") from exc
    return index, report

"""Offline chunking of a document already extracted by the PDF ingestion step."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from papertrail.ingestion.pdf_downloader import validate_paper_id
from papertrail.ingestion.pdf_extractor import ExtractedDocument
from papertrail.storage import atomic_write_bytes

from .chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, ChunkedDocument, ChunkingError, chunk_document, validate_chunk_settings

MAX_EXTRACTED_JSON_BYTES = 48 * 1024 * 1024


def chunk_saved_document(
    paper_id: str,
    *,
    data_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> tuple[ChunkedDocument, Path]:
    """Read local extracted JSON and atomically write a page-attributed chunk index.

    Does not download a PDF, read the network, create embeddings, or use an LLM.
    """
    paper_id = validate_paper_id(paper_id)
    validate_chunk_settings(chunk_size, overlap)
    stem = paper_id.replace("/", "__")
    extracted_path = data_dir / "extracted" / f"{stem}.json"
    output_path = data_dir / "chunks" / f"{stem}.json"
    if not extracted_path.is_file():
        raise ChunkingError(f"Extracted document not found: {extracted_path}. Run ingest-pdf first")
    try:
        size = extracted_path.stat().st_size
        if size > MAX_EXTRACTED_JSON_BYTES:
            raise ChunkingError("Extracted JSON exceeds allowed size")
        document = ExtractedDocument.model_validate_json(extracted_path.read_bytes())
    except ChunkingError:
        raise
    except (OSError, ValidationError, ValueError) as exc:
        raise ChunkingError("Extracted JSON is unreadable or invalid") from exc
    if document.paper_id != paper_id:
        raise ChunkingError("Extracted document paper ID does not match the requested ID")
    result = chunk_document(document, chunk_size=chunk_size, overlap=overlap)
    try:
        atomic_write_bytes(output_path, result.model_dump_json(indent=2).encode("utf-8"))
    except OSError as exc:
        raise ChunkingError("Unable to save chunk JSON") from exc
    return result, output_path

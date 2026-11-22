"""Deterministic page-aware character chunker with offsets into exact source text.

A chunk never crosses a page boundary. Neither tokenization nor an LLM is involved.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict, Field

from papertrail.ingestion.pdf_extractor import ExtractedDocument, ExtractedPage
from papertrail.ingestion.pdf_downloader import validate_paper_id

MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 8_000
DEFAULT_CHUNK_SIZE = 1_200
DEFAULT_OVERLAP = 200


class ChunkingError(ValueError):
    """Invalid input document or chunking settings."""


class TextChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_number: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str = Field(min_length=1)


class ChunkedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    source_url: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_pages: int = Field(ge=1)
    extracted_pages: int = Field(ge=1)
    truncated: bool
    chunk_size: int = Field(ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE)
    overlap: int = Field(ge=0)
    empty_text_pages: list[int]
    total_chunks: int = Field(ge=0)
    chunks: list[TextChunk]


def validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE:
        raise ChunkingError(f"chunk_size must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0 or overlap >= chunk_size // 2:
        raise ChunkingError("overlap must be nonnegative and less than half the chunk_size")


def _choose_end(text: str, start: int, chunk_size: int) -> int:
    hard_end = min(start + chunk_size, len(text))
    if hard_end == len(text):
        return hard_end

    # Prefer a paragraph, then a line, then a sentence or word boundary.
    # The lower bound avoids a pathological sequence of tiny chunks.
    lower = start + max(1, int(chunk_size * 0.65))
    paragraph = text.rfind("\n\n", lower, hard_end)
    if paragraph != -1:
        return paragraph + 2
    newline = text.rfind("\n", lower, hard_end)
    if newline != -1:
        return newline + 1
    matches = list(re.finditer(r"[.!?。！？][\s]+", text[lower:hard_end]))
    if matches:
        return lower + matches[-1].end()
    word = max(text.rfind(" ", lower, hard_end), text.rfind("\t", lower, hard_end))
    if word != -1:
        return word + 1
    return hard_end


def _page_spans(page: ExtractedPage, chunk_size: int, overlap: int) -> Iterator[tuple[int, int, str]]:
    text = page.text
    start = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start == len(text):
            break

        end = _choose_end(text, start, chunk_size)
        # Trim trailing whitespace but preserve *exact* character offsets.
        actual_end = end
        while actual_end > start and text[actual_end - 1].isspace():
            actual_end -= 1
        if actual_end > start:
            yield start, actual_end, text[start:actual_end]
        if end == len(text):
            break
        # We only trim whitespace when moving forward. Non-whitespace text is
        # never omitted: every new start is no later than the previous end.
        start = max(start + 1, actual_end - overlap)


def chunk_document(
    document: ExtractedDocument,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> ChunkedDocument:
    """Create deterministic chunks without modifying the extracted page text.

    end_char is exclusive, and ``page.text[start_char:end_char] == chunk.text``.
    """
    validate_chunk_settings(chunk_size, overlap)
    paper_id = validate_paper_id(document.paper_id)
    if document.source_url != f"https://arxiv.org/abs/{paper_id}":
        raise ChunkingError("Document source URL does not match arXiv paper ID")
    if not re.fullmatch(r"[0-9a-f]{64}", document.sha256):
        raise ChunkingError("Document SHA-256 must be a lowercase hexadecimal digest")
    if document.extracted_pages != len(document.pages) or document.total_pages < document.extracted_pages:
        raise ChunkingError("Document page counts are inconsistent")
    if [page.page_number for page in document.pages] != list(range(1, document.extracted_pages + 1)):
        raise ChunkingError("Extracted pages must be numbered consecutively from 1")

    chunks: list[TextChunk] = []
    for page in document.pages:
        if page.character_count != len(page.text) or page.has_text != bool(page.text):
            raise ChunkingError(f"Inconsistent page text metadata on page {page.page_number}")
        for start, end, content in _page_spans(page, chunk_size, overlap):
            fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
            chunks.append(TextChunk(
                chunk_id=f"{paper_id}@{document.sha256[:12]}:p{page.page_number:04d}:{start:06d}-{end:06d}:{fingerprint}",
                paper_id=paper_id,
                source_url=document.source_url,
                source_sha256=document.sha256,
                page_number=page.page_number,
                start_char=start,
                end_char=end,
                text=content,
            ))
    return ChunkedDocument(
        paper_id=paper_id,
        source_url=document.source_url,
        source_sha256=document.sha256,
        total_pages=document.total_pages,
        extracted_pages=document.extracted_pages,
        truncated=document.truncated,
        chunk_size=chunk_size,
        overlap=overlap,
        empty_text_pages=[page.page_number for page in document.pages if not page.text.strip()],
        total_chunks=len(chunks),
        chunks=chunks,
    )

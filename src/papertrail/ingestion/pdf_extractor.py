"""Page-aware text extraction. Images / scanned pages are explicitly flagged, not OCR'd."""

from __future__ import annotations

import hashlib
from typing import Literal

import pymupdf
from pydantic import BaseModel, ConfigDict, Field

from .pdf_downloader import validate_paper_id
from .layout import extract_page_text

MAX_PAGE_CHARS = 200_000


class PdfExtractionError(RuntimeError):
    """File is not a parseable or supported text PDF."""


class ExtractedPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str
    character_count: int = Field(ge=0)
    has_text: bool
    extraction_method: Literal["simple", "two_column"] = "simple"


class ExtractedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    source_url: str
    sha256: str
    total_pages: int = Field(ge=1)
    extracted_pages: int = Field(ge=1)
    truncated: bool
    empty_text_pages: list[int]
    pages: list[ExtractedPage] = Field(min_length=1)


def extract_pdf(pdf_bytes: bytes, paper_id: str, *, max_pages: int = 100, layout: str = "auto") -> ExtractedDocument:
    paper_id = validate_paper_id(paper_id)
    if not 1 <= max_pages <= 200:
        raise ValueError("max_pages must be between 1 and 200")
    if layout not in {"auto", "simple"}:
        raise ValueError("layout must be 'auto' or 'simple'")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise PdfExtractionError("PDF has an invalid signature")
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.needs_pass:
                raise PdfExtractionError("Password-protected PDFs are not supported")
            if doc.page_count < 1:
                raise PdfExtractionError("PDF has no pages")
            count = min(doc.page_count, max_pages)
            pages: list[ExtractedPage] = []
            for index in range(count):
                text, extraction_method = extract_page_text(doc[index], layout=layout)
                if len(text) > MAX_PAGE_CHARS:
                    raise PdfExtractionError(f"Page {index + 1} exceeds the extracted-text safety limit")
                pages.append(ExtractedPage(
                    page_number=index + 1,
                    text=text,
                    character_count=len(text),
                    has_text=bool(text),
                    extraction_method=extraction_method,
                ))
            return ExtractedDocument(
                paper_id=paper_id,
                source_url=f"https://arxiv.org/abs/{paper_id}",
                sha256=hashlib.sha256(pdf_bytes).hexdigest(),
                total_pages=doc.page_count,
                extracted_pages=count,
                truncated=count < doc.page_count,
                empty_text_pages=[p.page_number for p in pages if not p.has_text],
                pages=pages,
            )
    except PdfExtractionError:
        raise
    except (RuntimeError, ValueError, TypeError, pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise PdfExtractionError("PDF could not be parsed safely; file may be corrupt") from exc
